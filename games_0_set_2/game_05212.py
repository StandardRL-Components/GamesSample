
# Generated: 2025-08-28T04:19:03.838057
# Source Brief: brief_05212.md
# Brief Index: 5212

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect yellow gems and reach the green exit."
    )

    game_description = (
        "A minimalist pixel-art platformer. Navigate a procedurally generated level, "
        "collecting gems and reaching the exit as quickly as possible."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_PITFALL = (10, 10, 15)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_GEM = (255, 220, 50)
        self.COLOR_EXIT = (80, 255, 120)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_TEXT = (240, 240, 240)
        
        try:
            self.font_ui = pygame.font.Font(pygame.font.match_font('monospace'), 18)
        except:
            self.font_ui = pygame.font.SysFont('monospace', 18)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.GRAVITY = 0.5
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.12
        self.JUMP_STRENGTH = -10
        self.PLAYER_SIZE = 20
        self.MAX_FALL_SPEED = 12
        self.MAX_RUN_SPEED = 6

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.gems = None
        self.exit_rect = None
        self.particles = None
        self.camera_x = None
        self.world_width = None
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def _generate_level(self):
        self.platforms = []
        self.gems = []
        
        # Starting platform
        start_plat_width = 200
        start_y = 350
        self.platforms.append(pygame.Rect(0, start_y, start_plat_width, self.screen_height - start_y))
        
        current_x = start_plat_width
        current_y = start_y
        
        # Procedurally generate level segments
        for _ in range(20):
            gap_width = self.np_random.integers(60, 120)
            platform_width = self.np_random.integers(150, 400)
            
            # Ensure jump is possible
            max_y_change = int(abs(gap_width * 1.2)) # Heuristic for jumpable height
            y_change = self.np_random.integers(-max_y_change, max_y_change)
            
            next_y = np.clip(current_y + y_change, 150, 370)
            
            current_x += gap_width
            
            new_platform = pygame.Rect(current_x, next_y, platform_width, self.screen_height - next_y)
            self.platforms.append(new_platform)
            
            # Add gems on the platform
            for i in range(self.np_random.integers(1, 4)):
                gem_x = current_x + (i + 1) * platform_width / 4 + self.np_random.integers(-10, 10)
                gem_y = next_y - 30 - self.np_random.integers(0, 40)
                self.gems.append(pygame.Rect(int(gem_x), int(gem_y), 12, 12))

            current_x += platform_width
            current_y = next_y

        # Place exit on the last platform
        last_plat = self.platforms[-1]
        self.exit_rect = pygame.Rect(last_plat.centerx - 15, last_plat.top - 30, 30, 30)
        self.world_width = last_plat.right + 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        self.player_pos = pygame.math.Vector2(100, 200)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False
        
        self.particles = []
        self.camera_x = 0
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        is_moving_horizontally = False
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            is_moving_horizontally = True
        if movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
            is_moving_horizontally = True
        
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # --- Physics Update ---
        # Friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        
        # Speed limits
        self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_RUN_SPEED, self.MAX_RUN_SPEED)
        self.player_vel.y = np.clip(self.player_vel.y, -float('inf'), self.MAX_FALL_SPEED)
        
        # Gravity
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.on_ground = False
        
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check for vertical collision (landing on top)
                if self.player_vel.y > 0 and player_rect.bottom - self.player_vel.y <= plat.top:
                    player_rect.bottom = plat.top
                    self.player_pos.y = player_rect.y
                    self.player_vel.y = 0
                    self.on_ground = True
                # Check for horizontal collision
                elif self.player_vel.x > 0 and player_rect.right - self.player_vel.x <= plat.left:
                    player_rect.right = plat.left
                    self.player_pos.x = player_rect.x
                    self.player_vel.x = 0
                elif self.player_vel.x < 0 and player_rect.left - self.player_vel.x >= plat.right:
                    player_rect.left = plat.right
                    self.player_pos.x = player_rect.x
                    self.player_vel.x = 0
                # Check for hitting ceiling
                elif self.player_vel.y < 0 and player_rect.top - self.player_vel.y >= plat.bottom:
                    player_rect.top = plat.bottom
                    self.player_pos.y = player_rect.y
                    self.player_vel.y = 0

        # --- Event Handling & Rewards ---
        reward = 0
        terminated = False
        
        # Survival reward
        reward += 0.01
        
        # Penalty for standing still
        if not is_moving_horizontally and self.on_ground:
            reward -= 0.02
        
        # Gem collection
        collected_indices = player_rect.collidelistall(self.gems)
        if collected_indices:
            for i in sorted(collected_indices, reverse=True):
                gem_pos = self.gems[i].center
                for _ in range(15): # Particle burst
                    self.particles.append([list(gem_pos), [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)], self.np_random.integers(10, 20)])
                del self.gems[i]
                self.score += 10
                reward += 1
                self.gems_collected += 1
                # sfx: gem_collect
        
        # Reaching exit
        if player_rect.colliderect(self.exit_rect):
            self.score += 50
            reward += 50
            time_bonus = max(0, 20 * (1 - self.steps / self.MAX_STEPS))
            self.score += time_bonus
            reward += time_bonus
            terminated = True
            self.game_over = True
            # sfx: level_complete
        
        # Falling into a pit
        if self.player_pos.y > self.screen_height + self.PLAYER_SIZE:
            self.score -= 10
            reward -= 10
            terminated = True
            self.game_over = True
            # sfx: fall_death

        # Max steps termination
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if self.auto_advance:
            self.clock.tick(30)

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[2] -= 1 # lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        # Update camera to follow player
        self.camera_x = self.player_pos.x - self.screen_width / 2
        self.camera_x = np.clip(self.camera_x, 0, self.world_width - self.screen_width)

        # Update visual effects
        self._update_particles()
        
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements (with camera offset) ---
        # Draw pitfalls (as part of background)
        for plat in self.platforms:
            pit_y = plat.bottom
            pit_height = self.screen_height - pit_y
            if pit_height > 0:
                pygame.draw.rect(self.screen, self.COLOR_PITFALL, (int(plat.x - self.camera_x), pit_y, plat.width, pit_height))
        
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (int(plat.x - self.camera_x), plat.y, plat.width, plat.height))
        
        # Draw gems
        for gem in self.gems:
            pygame.gfxdraw.box(self.screen, (int(gem.x - self.camera_x), gem.y, gem.width, gem.height), self.COLOR_GEM)

        # Draw exit
        pygame.gfxdraw.box(self.screen, (int(self.exit_rect.x - self.camera_x), self.exit_rect.y, self.exit_rect.width, self.exit_rect.height), self.COLOR_EXIT)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p[0][0] - self.camera_x), int(p[0][1])), int(p[2] / 4))
        
        # Draw player
        player_rect_on_screen = pygame.Rect(int(self.player_pos.x - self.camera_x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_on_screen)
        if not self.on_ground: # Simple jumping animation
            pygame.draw.rect(self.screen, (255,255,255), player_rect_on_screen.inflate(-8, -8))


        # --- Render UI Overlay ---
        time_text = f"TIME: {self.steps / 30:.1f}s"
        gem_text = f"GEMS: {self.gems_collected}"
        
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        gem_surf = self.font_ui.render(gem_text, True, self.COLOR_GEM)
        
        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(gem_surf, (self.screen_width - gem_surf.get_width() - 10, 10))
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Pixel Platformer")
    
    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # No down action in this game
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it.
        # Need to transpose it back for pygame's display format.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()