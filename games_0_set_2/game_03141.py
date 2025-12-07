
# Generated: 2025-08-28T07:05:38.535572
# Source Brief: brief_03141.md
# Brief Index: 3141

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship. "
        "Collect the yellow gems and avoid the moving blocks."
    )

    # User-facing game description
    game_description = (
        "A fast-paced arcade game. Collect 25 shimmering gems to win, "
        "but watch out for the deadly obstacles! The game ends if you collide "
        "with an obstacle or the 60-second timer runs out."
    )

    # Frames auto-advance at 30 FPS
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIMER_SECONDS = 60

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (255, 220, 0, 60)
    OBSTACLE_COLORS = [(255, 80, 80), (255, 140, 80), (180, 80, 255)] # Red, Orange, Purple
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game parameters
    PLAYER_SIZE = 16
    PLAYER_SPEED = 5
    GEM_SIZE = 12
    OBSTACLE_SIZE = 24
    NUM_OBSTACLES = 12
    GEMS_TO_WIN = 25
    
    INITIAL_OBSTACLE_SPEED = 1.5
    OBSTACLE_SPEED_INCREASE_INTERVAL = 10 * FPS # every 10 seconds
    OBSTACLE_SPEED_INCREASE_AMOUNT = 0.25

    PROXIMITY_THRESHOLD = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.gems = None
        self.obstacles = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.gem_count = 0
        self.time_left = 0
        self.current_obstacle_speed = 0
        self.game_over = False
        self.game_outcome = ""

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self._spawn_obstacle()

        self.gems = []
        self._spawn_gem()
            
        self.particles = []
        self.steps = 0
        self.score = 0
        self.gem_count = 0
        self.time_left = self.TIMER_SECONDS * self.FPS
        self.current_obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.game_over = False
        self.game_outcome = ""
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # --- Update Game Logic ---
            
            # 1. Handle player movement and proximity rewards
            reward += self._handle_player_movement(movement)
            
            # 2. Update obstacles
            self._update_obstacles()
            
            # 3. Update particles
            self._update_particles()
            
            # 4. Handle collisions
            reward += self._handle_collisions()
            
            # 5. Update timer and difficulty
            self.time_left -= 1
            if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
                self.current_obstacle_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT
            
            # 6. Survival reward
            reward += 0.01

        # --- Check Termination Conditions ---
        if self.gem_count >= self.GEMS_TO_WIN and not self.game_over:
            self.game_over = True
            self.game_outcome = "YOU WIN!"
            reward += 100
        
        if self.time_left <= 0 and not self.game_over:
            self.game_over = True
            self.game_outcome = "TIME UP"
            reward -= 10

        if self.game_over:
            terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        # Keep player within bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
        
        # Proximity reward/penalty
        proximity_reward = 0
        if move_vec.length() > 0:
            min_dist = float('inf')
            closest_obs = None
            for obs in self.obstacles:
                dist = self.player_pos.distance_to(obs['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_obs = obs
            
            if closest_obs and min_dist < self.PROXIMITY_THRESHOLD:
                vec_to_obs = (closest_obs['pos'] - self.player_pos).normalize()
                dot_product = move_vec.dot(vec_to_obs)
                if dot_product > 0.5: # Moving towards obstacle
                    proximity_reward = -1.0
                elif dot_product < -0.5: # Moving away from obstacle
                    proximity_reward = 2.0
        return proximity_reward

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'] += obs['vel'] * self.current_obstacle_speed
            # Screen wrap-around
            if obs['pos'].x < -self.OBSTACLE_SIZE: obs['pos'].x = self.WIDTH + self.OBSTACLE_SIZE
            if obs['pos'].x > self.WIDTH + self.OBSTACLE_SIZE: obs['pos'].x = -self.OBSTACLE_SIZE
            if obs['pos'].y < -self.OBSTACLE_SIZE: obs['pos'].y = self.HEIGHT + self.OBSTACLE_SIZE
            if obs['pos'].y > self.HEIGHT + self.OBSTACLE_SIZE: obs['pos'].y = -self.OBSTACLE_SIZE
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95 # Damping

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player vs Gems
        for gem in self.gems[:]:
            gem_rect = pygame.Rect(gem['pos'].x - self.GEM_SIZE / 2, gem['pos'].y - self.GEM_SIZE / 2, self.GEM_SIZE, self.GEM_SIZE)
            if player_rect.colliderect(gem_rect):
                # SFX: Gem collect sound
                self.gems.remove(gem)
                self.gem_count += 1
                self.score += 10
                reward += 10
                self._spawn_gem_particles(gem['pos'])
                if self.gem_count < self.GEMS_TO_WIN:
                    self._spawn_gem()
                break

        # Player vs Obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x - self.OBSTACLE_SIZE / 2, obs['pos'].y - self.OBSTACLE_SIZE / 2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            if player_rect.colliderect(obs_rect):
                # SFX: Explosion sound
                self.game_over = True
                self.game_outcome = "GAME OVER"
                reward -= 100
                break
        return reward

    def _spawn_gem(self):
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(self.GEM_SIZE, self.WIDTH - self.GEM_SIZE),
                self.np_random.uniform(self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE)
            )
            # Ensure gem doesn't spawn on an obstacle
            is_safe = True
            for obs in self.obstacles:
                if pos.distance_to(obs['pos']) < self.OBSTACLE_SIZE * 2:
                    is_safe = False
                    break
            if is_safe:
                self.gems.append({'pos': pos})
                break

    def _spawn_obstacle(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.OBSTACLE_SIZE)
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.OBSTACLE_SIZE)
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 2: # Left
            pos = pygame.Vector2(-self.OBSTACLE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else: # Right
            pos = pygame.Vector2(self.WIDTH + self.OBSTACLE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)

        vel = pygame.Vector2(math.cos(angle), math.sin(angle))
        color = self.np_random.choice(self.OBSTACLE_COLORS)
        self.obstacles.append({'pos': pos, 'vel': vel, 'color': color})

    def _spawn_gem_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, obs['color'], (obs['pos'].x - self.OBSTACLE_SIZE / 2, obs['pos'].y - self.OBSTACLE_SIZE / 2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE))

        # Render gems with pulsating glow
        for gem in self.gems:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            glow_size = self.GEM_SIZE * (1.5 + pulse * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(gem['pos'].x), int(gem['pos'].y), int(glow_size), self.COLOR_GEM_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(gem['pos'].x), int(gem['pos'].y), int(self.GEM_SIZE/2), self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, int(gem['pos'].x), int(gem['pos'].y), int(self.GEM_SIZE/2), self.COLOR_GEM)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30.0))
            color = (*self.COLOR_GEM[:3], alpha)
            size = max(1, int(p['life'] / 10))
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)
            
        # Render player with glow
        if not (self.game_over and self.game_outcome == "GAME OVER"):
            glow_size = self.PLAYER_SIZE * 1.8
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(glow_size), self.COLOR_PLAYER_GLOW)
            player_rect = (self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Gems collected
        gem_text = self.font_ui.render(f"GEMS: {self.gem_count}/{self.GEMS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Time left
        time_str = f"{self.time_left // self.FPS:02d}"
        time_text = self.font_ui.render(f"TIME: {time_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            outcome_text = self.font_game_over.render(self.game_outcome, True, self.COLOR_UI_TEXT)
            text_rect = outcome_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gem_count": self.gem_count,
            "time_left_seconds": self.time_left // self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display for testing
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    print("\n" + GameEnv.user_guide + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Map keyboard inputs to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            pass

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()