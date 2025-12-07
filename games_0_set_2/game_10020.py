import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:40:13.028764
# Source Brief: brief_00020.md
# Brief Index: 20
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls a size-shifting ball.
    The goal is to collect 10 large green gems while avoiding 20 small red gems
    before a 60-second timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a size-shifting ball to collect green gems while avoiding red ones before time runs out."
    user_guide = "Use ↑↓←→ arrow keys to move. Press space to shift between small/fast and large/powerful forms."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30 # For game logic, not rendering speed
    MAX_STEPS = 60 * FPS # 60 seconds

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_LARGE_GEM = (0, 255, 150)
    COLOR_SMALL_GEM = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)

    # Player settings
    PLAYER_SIZE_SMALL = 10
    PLAYER_SIZE_LARGE = 20
    PLAYER_ACCEL_SMALL = 0.8
    PLAYER_ACCEL_LARGE = 0.5 # Slower accel when large for 'heavy' feel
    PLAYER_MAX_SPEED_SMALL = 6.0
    PLAYER_MAX_SPEED_LARGE = 8.0 # Faster top speed when large
    PLAYER_DRAG = 0.92
    PLAYER_SIZE_CHANGE_COOLDOWN = 0.5 * FPS

    # Gem settings
    NUM_LARGE_GEMS = 10
    NUM_SMALL_GEMS = 25 # Spawn more than needed to lose, for variety
    LARGE_GEM_SIZE = 8
    SMALL_GEM_SIZE = 5
    WIN_CONDITION_GEMS = 10
    LOSE_CONDITION_GEMS = 20
    
    # --- Helper Classes ---
    class Particle:
        def __init__(self, x, y, color, speed, life):
            self.pos = pygame.Vector2(x, y)
            self.color = color
            self.life = life
            self.initial_life = life
            angle = random.uniform(0, 2 * math.pi)
            self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * random.uniform(0.5, 1.5)

        def update(self):
            self.pos += self.vel
            self.vel *= 0.95 # Drag
            self.life -= 1

        def draw(self, surface):
            if self.life > 0:
                alpha = int(255 * (self.life / self.initial_life))
                radius = int(2 * (self.life / self.initial_life) + 1)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos.x - radius), int(self.pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_end = pygame.font.SysFont("Verdana", 50, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.player_target_size = None
        self.size_change_cooldown = 0
        self.prev_space_held = False
        self.large_gems = []
        self.small_gems = []
        self.particles = []
        self.large_gems_collected = 0
        self.small_gems_hit = 0
        self.prev_dist_to_large_gem = float('inf')
        self.prev_dist_to_small_gem = float('inf')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = self.PLAYER_SIZE_SMALL
        self.player_target_size = self.PLAYER_SIZE_SMALL
        self.size_change_cooldown = 0
        self.prev_space_held = False
        
        self.large_gems_collected = 0
        self.small_gems_hit = 0
        
        self.particles.clear()
        self._spawn_gems()
        
        self.prev_dist_to_large_gem = self._get_closest_gem_dist(self.large_gems)
        self.prev_dist_to_small_gem = self._get_closest_gem_dist(self.small_gems)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- Handle Input and Update Player ---
        self._handle_input(action)
        self._update_player()

        # --- Update Game Objects ---
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        # --- Collisions and Events ---
        reward += self._check_collisions()
        
        # --- Continuous Rewards ---
        dist_large = self._get_closest_gem_dist(self.large_gems)
        if dist_large < self.prev_dist_to_large_gem:
            reward += 0.01 # Small reward for getting closer
        self.prev_dist_to_large_gem = dist_large

        dist_small = self._get_closest_gem_dist(self.small_gems)
        if dist_small > self.prev_dist_to_small_gem:
            reward += 0.005 # Tiny reward for moving away from danger
        self.prev_dist_to_small_gem = dist_small
        
        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        accel = self.PLAYER_ACCEL_LARGE if self.player_size > self.PLAYER_SIZE_SMALL else self.PLAYER_ACCEL_SMALL
        
        if movement == 1: self.player_vel.y -= accel # Up
        if movement == 2: self.player_vel.y += accel # Down
        if movement == 3: self.player_vel.x -= accel # Left
        if movement == 4: self.player_vel.x += accel # Right
        
        # Handle size change on button press (not hold)
        if space_held and not self.prev_space_held and self.size_change_cooldown <= 0:
            self.size_change_cooldown = self.PLAYER_SIZE_CHANGE_COOLDOWN
            if self.player_target_size == self.PLAYER_SIZE_SMALL:
                self.player_target_size = self.PLAYER_SIZE_LARGE
            else:
                self.player_target_size = self.PLAYER_SIZE_SMALL
            # SFX: Size change sound
            self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 10, 2.0, 20)
        
        self.prev_space_held = space_held
        if self.size_change_cooldown > 0:
            self.size_change_cooldown -= 1

    def _update_player(self):
        # Smooth size transition
        if abs(self.player_size - self.player_target_size) > 0.1:
            self.player_size += (self.player_target_size - self.player_size) * 0.2
        else:
            self.player_size = self.player_target_size
            
        # Limit speed based on size
        max_speed = self.PLAYER_MAX_SPEED_LARGE if self.player_size > self.PLAYER_SIZE_SMALL else self.PLAYER_MAX_SPEED_SMALL
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
            
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Update position
        self.player_pos += self.player_vel
        
        # Wall bouncing
        if self.player_pos.x - self.player_size < 0:
            self.player_pos.x = self.player_size
            self.player_vel.x *= -0.8 # Dampen bounce
            # SFX: Bounce sound
        elif self.player_pos.x + self.player_size > self.SCREEN_WIDTH:
            self.player_pos.x = self.SCREEN_WIDTH - self.player_size
            self.player_vel.x *= -0.8
            # SFX: Bounce sound
        if self.player_pos.y - self.player_size < 0:
            self.player_pos.y = self.player_size
            self.player_vel.y *= -0.8
            # SFX: Bounce sound
        elif self.player_pos.y + self.player_size > self.SCREEN_HEIGHT:
            self.player_pos.y = self.SCREEN_HEIGHT - self.player_size
            self.player_vel.y *= -0.8
            # SFX: Bounce sound

    def _spawn_particles(self, pos, color, count, speed, life):
        for _ in range(count):
            self.particles.append(self.Particle(pos.x, pos.y, color, speed, life))

    def _spawn_gems(self):
        self.large_gems.clear()
        self.small_gems.clear()
        
        # Create a grid to avoid spawning gems on top of each other
        cell_size = 50
        grid_w, grid_h = self.SCREEN_WIDTH // cell_size, self.SCREEN_HEIGHT // cell_size
        occupied_cells = set()
        
        # Avoid spawning near center
        center_x, center_y = grid_w // 2, grid_h // 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                occupied_cells.add((center_x + i, center_y + j))

        def get_random_pos():
            while True:
                cell_x = self.np_random.integers(0, grid_w)
                cell_y = self.np_random.integers(0, grid_h)
                if (cell_x, cell_y) not in occupied_cells:
                    occupied_cells.add((cell_x, cell_y))
                    x = cell_x * cell_size + self.np_random.uniform(10, cell_size - 10)
                    y = cell_y * cell_size + self.np_random.uniform(10, cell_size - 10)
                    return pygame.Vector2(x, y)

        for _ in range(self.NUM_LARGE_GEMS):
            self.large_gems.append(get_random_pos())
        for _ in range(self.NUM_SMALL_GEMS):
            self.small_gems.append(get_random_pos())
    
    def _check_collisions(self):
        reward = 0
        
        # Large gems
        for gem_pos in self.large_gems[:]:
            if self.player_pos.distance_to(gem_pos) < self.player_size + self.LARGE_GEM_SIZE:
                self.large_gems.remove(gem_pos)
                self.large_gems_collected += 1
                reward += 10
                # SFX: Large gem collect
                self._spawn_particles(gem_pos, self.COLOR_LARGE_GEM, 30, 3.0, 30)
                if len(self.large_gems) == 0: # Respawn if all collected before win condition
                    self._spawn_gems()

        # Small gems
        for gem_pos in self.small_gems[:]:
            if self.player_pos.distance_to(gem_pos) < self.player_size + self.SMALL_GEM_SIZE:
                self.small_gems.remove(gem_pos)
                self.small_gems_hit += 1
                reward -= 5
                # SFX: Small gem hit
                self._spawn_particles(gem_pos, self.COLOR_SMALL_GEM, 15, 2.0, 20)

        return reward

    def _check_termination(self):
        if self.large_gems_collected >= self.WIN_CONDITION_GEMS:
            self.game_over = True
            return True, 100 # Win
        if self.small_gems_hit >= self.LOSE_CONDITION_GEMS:
            self.game_over = True
            return True, -100 # Lose by collision
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, -50 # Lose by timeout
        return False, 0
    
    def _get_closest_gem_dist(self, gem_list):
        if not gem_list:
            return float('inf')
        return min(self.player_pos.distance_to(gem) for gem in gem_list)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Render gems
        self._render_gems(self.large_gems, self.COLOR_LARGE_GEM, self.LARGE_GEM_SIZE)
        self._render_gems(self.small_gems, self.COLOR_SMALL_GEM, self.SMALL_GEM_SIZE)

        # Render player
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        size = int(self.player_size)
        # Glow effect
        for i in range(size // 2, 0, -2):
            alpha = 80 - (i * (80 // (size // 2 + 1)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + i, (*self.COLOR_PLAYER, alpha))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)

    def _render_gems(self, gem_list, color, radius):
        for pos in gem_list:
            ipos = (int(pos.x), int(pos.y))
            # Glow effect
            for i in range(radius, 0, -1):
                alpha = 100 - (i * (100 // (radius + 1)))
                pygame.gfxdraw.filled_circle(self.screen, ipos[0], ipos[1], radius + i, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, ipos[0], ipos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, ipos[0], ipos[1], radius, color)

    def _render_ui(self):
        # Large Gems Collected
        large_gem_text = self.font_ui.render(f"GEMS: {self.large_gems_collected}/{self.WIN_CONDITION_GEMS}", True, self.COLOR_LARGE_GEM)
        self.screen.blit(large_gem_text, (10, 10))
        
        # Small Gems Hit
        small_gem_text = self.font_ui.render(f"HITS: {self.small_gems_hit}/{self.LOSE_CONDITION_GEMS}", True, self.COLOR_SMALL_GEM)
        self.screen.blit(small_gem_text, (10, 40))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text_str = f"{max(0, time_left):.1f}"
        timer_color = self.COLOR_UI_TEXT
        if time_left < 10: timer_color = self.COLOR_TIMER_CRIT
        elif time_left < 20: timer_color = self.COLOR_TIMER_WARN
        time_text = self.font_ui.render(time_text_str, True, timer_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # End game message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            if self.large_gems_collected >= self.WIN_CONDITION_GEMS:
                end_text_str, color = "VICTORY!", self.COLOR_LARGE_GEM
            elif self.small_gems_hit >= self.LOSE_CONDITION_GEMS:
                end_text_str, color = "GAME OVER", self.COLOR_SMALL_GEM
            else: # Timeout
                end_text_str, color = "TIME UP", self.COLOR_TIMER_WARN
                
            end_text = self.font_end.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "large_gems_collected": self.large_gems_collected,
            "small_gems_hit": self.small_gems_hit,
            "time_left_seconds": (self.MAX_STEPS - self.steps) / self.FPS
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # This method is for developer convenience and is not required by Gymnasium.
        # It's safe to remove if it causes issues.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            test_obs = self._get_observation()
            assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            # Test reset
            obs, info = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert not trunc
            assert isinstance(info, dict)
        except Exception as e:
            # This helps in debugging during development.
            print(f"Validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, ensure you have pygame installed: pip install pygame
    # Un-comment the line below to run with a visible window
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    # The validation call in __init__ is removed from the main execution path
    # to avoid printing validation messages when running the game.
    # We can do this by creating a separate instance or modifying the class.
    # For simplicity, we just accept the print statement on first run.
    # env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # The original code had a validation method that printed on init.
    # To avoid this, we can remove the call from __init__ or just ignore it.
    # The code below is the manual play loop.
    try:
        env.validate_implementation()
    except:
        pass # Ignore validation errors in manual play mode

    while not done:
        # --- Action mapping for human player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering for human player ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling and clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    
    # Keep the window open for a few seconds to see the end screen
    pygame.time.wait(3000)
    
    env.close()