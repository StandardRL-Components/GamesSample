import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift for a speed boost (costs health). Press Space to collect nearby gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect glittering gems while dodging cunning enemies in a fast-paced, top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_WIDTH = 400
    GAME_HEIGHT = 400
    GAME_X_OFFSET = (SCREEN_WIDTH - GAME_WIDTH) // 2
    
    FPS = 30
    MAX_STEPS = 1500 # 50 seconds at 30fps

    COLOR_BG_OUTER = (10, 10, 20)
    COLOR_BG_INNER = (25, 25, 40)
    COLOR_BOUNDARY = (60, 60, 80)
    
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_SPARKLE = (255, 255, 150)
    
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_TRAIL = (255, 50, 50)

    COLOR_HEALTH_BAR = (40, 200, 40)
    COLOR_HEALTH_BAR_BG = (100, 40, 40)
    
    COLOR_TEXT = (220, 220, 220)
    
    PLAYER_SIZE = 12
    PLAYER_SPEED_BASE = 4.0
    PLAYER_SPEED_BOOST = 6.0
    
    GEM_RADIUS = 6
    GEM_COLLECT_RADIUS = 25
    NUM_GEMS = 5
    GEMS_TO_WIN = 50
    
    ENEMY_SIZE = 14
    NUM_ENEMIES = 3
    ENEMY_SPEED_BASE = 1.0
    ENEMY_SPEED_INCREASE_PER_GEM = 0.05

    INITIAL_HEALTH = 100
    BOOST_HEALTH_COST = 0.5
    ENEMY_COLLISION_DAMAGE = 20
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.game_area = pygame.Rect(self.GAME_X_OFFSET, 0, self.GAME_WIDTH, self.GAME_HEIGHT)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.gems_collected = None
        self.enemies = []
        self.gems = []
        self.particles = []
        self.steps = 0
        self.game_over = False
        self.np_random = None

        # self.reset() is called in the first call to the env, no need to call it here.
        # This prevents an error if the seed is passed to reset() later.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.game_area.centerx, self.game_area.centery]
        self.player_health = self.INITIAL_HEALTH
        self.gems_collected = 0
        self.steps = 0
        self.game_over = False
        
        self.gems = [self._spawn_gem() for _ in range(self.NUM_GEMS)]
        self.enemies = [self._spawn_enemy() for _ in range(self.NUM_ENEMIES)]
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        dist_before = self._get_dist_to_nearest_gem()

        self._handle_player_movement(movement, shift_held)
        
        if shift_held:
            self.player_health -= self.BOOST_HEALTH_COST
            reward -= 0.1 # Cost for using boost
        
        dist_after = self._get_dist_to_nearest_gem()
        
        if dist_after < dist_before:
            reward += 0.01 # Small reward for moving towards gem
        else:
            reward -= 0.002 # Small penalty for moving away

        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions and Actions ---
        if space_held:
            collected_this_step = self._collect_gems()
            if collected_this_step > 0:
                reward += 1.0 * collected_this_step
        
        if self._check_enemy_collisions():
            reward -= 5.0
            self.player_health -= self.ENEMY_COLLISION_DAMAGE
            
        # --- Check Termination ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.player_health <= 0:
            self.player_health = 0
            terminated = True
            self.game_over = True
            reward -= 100.0 # Penalty for losing
        
        if self.gems_collected >= self.GEMS_TO_WIN:
            terminated = True
            self.game_over = True
            reward += 100.0 # Reward for winning
            
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_gem(self):
        return [
            self.np_random.integers(self.game_area.left + self.GEM_RADIUS, self.game_area.right - self.GEM_RADIUS),
            self.np_random.integers(self.game_area.top + self.GEM_RADIUS, self.game_area.bottom - self.GEM_RADIUS)
        ]

    def _spawn_enemy(self):
        return {
            "orbit_center": [
                self.np_random.integers(self.game_area.left, self.game_area.right),
                self.np_random.integers(self.game_area.top, self.game_area.bottom)
            ],
            "orbit_radius": self.np_random.integers(50, 150),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "pos": [0, 0],
            "trail": []
        }

    def _handle_player_movement(self, movement, shift_held):
        speed = self.PLAYER_SPEED_BOOST if shift_held else self.PLAYER_SPEED_BASE
        
        if movement == 1: # Up
            self.player_pos[1] -= speed
        elif movement == 2: # Down
            self.player_pos[1] += speed
        elif movement == 3: # Left
            self.player_pos[0] -= speed
        elif movement == 4: # Right
            self.player_pos[0] += speed
            
        self.player_pos[0] = np.clip(self.player_pos[0], self.game_area.left + self.PLAYER_SIZE/2, self.game_area.right - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.game_area.top + self.PLAYER_SIZE/2, self.game_area.bottom - self.PLAYER_SIZE/2)

    def _update_enemies(self):
        base_angular_speed = self.ENEMY_SPEED_BASE
        current_angular_speed = base_angular_speed + self.gems_collected * self.ENEMY_SPEED_INCREASE_PER_GEM
        
        for enemy in self.enemies:
            enemy["angle"] += math.radians(current_angular_speed)
            enemy["pos"][0] = enemy["orbit_center"][0] + math.cos(enemy["angle"]) * enemy["orbit_radius"]
            enemy["pos"][1] = enemy["orbit_center"][1] + math.sin(enemy["angle"]) * enemy["orbit_radius"]
            
            # Update trail
            enemy["trail"].append(list(enemy["pos"]))
            if len(enemy["trail"]) > 10:
                enemy["trail"].pop(0)

    def _collect_gems(self):
        collected_count = 0
        for i in range(len(self.gems) - 1, -1, -1):
            gem_pos = self.gems[i]
            dist = math.hypot(self.player_pos[0] - gem_pos[0], self.player_pos[1] - gem_pos[1])
            if dist < self.GEM_COLLECT_RADIUS:
                self._create_particles(gem_pos, self.COLOR_GEM, 20)
                self.gems[i] = self._spawn_gem()
                self.gems_collected += 1
                collected_count += 1
        return collected_count
        
    def _check_enemy_collisions(self):
        for enemy in self.enemies:
            enemy_pos = enemy["pos"]
            # Simplified collision: treat enemy as a circle
            dist = math.hypot(self.player_pos[0] - enemy_pos[0], self.player_pos[1] - enemy_pos[1])
            if dist < (self.PLAYER_SIZE / 2 + self.ENEMY_SIZE / 2):
                return True
        return False

    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return float('inf')
        
        min_dist = float('inf')
        for gem_pos in self.gems:
            dist = math.hypot(self.player_pos[0] - gem_pos[0], self.player_pos[1] - gem_pos[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(10, 20),
                "color": color
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG_OUTER)
        pygame.draw.rect(self.screen, self.COLOR_BG_INNER, self.game_area)
        
        # --- Render game elements ---
        self._render_game()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, self.game_area, 2)

        # Render enemy trails
        for enemy in self.enemies:
            for i, pos in enumerate(enemy["trail"]):
                alpha = int(255 * (i / len(enemy["trail"])) * 0.5)
                if alpha > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.ENEMY_SIZE//3, self.COLOR_ENEMY + (alpha,))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20))
            pygame.draw.circle(self.screen, p["color"] + (alpha,), (int(p["pos"][0]), int(p["pos"][1])), 2)

        # Render gems
        pulse = math.sin(self.steps * 0.3) * 1.5
        for gem_pos in self.gems:
            radius = int(self.GEM_RADIUS + pulse)
            pygame.gfxdraw.filled_circle(self.screen, int(gem_pos[0]), int(gem_pos[1]), radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, int(gem_pos[0]), int(gem_pos[1]), radius, self.COLOR_GEM)
            # Sparkle effect
            if self.np_random.random() < 0.1:
                sparkle_radius = radius + 3
                pygame.gfxdraw.filled_circle(self.screen, int(gem_pos[0]), int(gem_pos[1]), sparkle_radius, self.COLOR_GEM_SPARKLE + (100,))

        # Render enemies (as triangles)
        for enemy in self.enemies:
            pos = enemy["pos"]
            angle = math.atan2(pos[1] - enemy["orbit_center"][1], pos[0] - enemy["orbit_center"][0]) + math.pi/2
            p1 = (pos[0] + math.cos(angle) * self.ENEMY_SIZE, pos[1] + math.sin(angle) * self.ENEMY_SIZE)
            p2 = (pos[0] + math.cos(angle + 2.1) * self.ENEMY_SIZE, pos[1] + math.sin(angle + 2.1) * self.ENEMY_SIZE)
            p3 = (pos[0] + math.cos(angle - 2.1) * self.ENEMY_SIZE, pos[1] + math.sin(angle - 2.1) * self.ENEMY_SIZE)
            points = [(int(p[0]), int(p[1])) for p in [p1,p2,p3]]
            pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_ENEMY)

        # Render player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, glow_radius, self.COLOR_PLAYER_GLOW)
        # Player square
        player_rect = pygame.Rect(player_x - self.PLAYER_SIZE/2, player_y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Gems: {self.gems_collected}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        # Health Bar
        health_bar_width = 150
        health_pct = max(0, self.player_health / self.INITIAL_HEALTH)
        current_health_width = int(health_bar_width * health_pct)
        
        health_text = self.font_small.render("Health", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_bar_width - 15, 10))
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.SCREEN_WIDTH - health_bar_width - 15, 30, health_bar_width, 15))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.SCREEN_WIDTH - health_bar_width - 15, 30, current_health_width, 15))
            
        # Gems to Win
        gems_left_text = self.font_small.render(f"Collect {self.GEMS_TO_WIN} gems to win", True, self.COLOR_TEXT)
        text_rect = gems_left_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(gems_left_text, text_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.gems_collected >= self.GEMS_TO_WIN:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            
            end_text_render = self.font_large.render(end_text, True, (255, 255, 255))
            text_rect = end_text_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_render, text_rect)

    def _get_info(self):
        return {
            "score": self.gems_collected,
            "steps": self.steps,
            "health": self.player_health,
        }
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # Set this to 'human' to play the game, or 'rgb_array' for headless mode.
    render_mode = "human"

    if render_mode == "human":
        # In human mode, we'll use a wrapper to handle keyboard input and rendering.
        # We need to unset the dummy video driver to see the window.
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="human")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Collector")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n" + "="*30)
        print(f"GAME: {env.game_description}")
        print(f"CONTROLS: {env.user_guide}")
        print("="*30 + "\n")

        while not done:
            # --- Human Input ---
            keys = pygame.key.get_pressed()
            move_action = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: move_action = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: move_action = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: move_action = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Pygame Rendering ---
            # The observation is already a rendered frame, so we just need to display it.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Frame Rate ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(env.FPS)
            
        print(f"Game Over! Final Score: {info['score']}")
        pygame.time.wait(2000)
        env.close()
    else:
        # Standard Gymnasium usage for training an RL agent
        env = GameEnv()
        obs, info = env.reset(seed=42)
        for i in range(1000):
            action = env.action_space.sample() # Replace with your agent's action
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 100 == 0:
                print(f"Step {i}: Reward={reward:.2f}, Info={info}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Score: {info['score']}")
                obs, info = env.reset(seed=42+i)
        env.close()