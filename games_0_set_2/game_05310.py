
# Generated: 2025-08-28T04:36:40.776918
# Source Brief: brief_05310.md
# Brief Index: 5310

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move, ↑ to jump."

    # Must be a short, user-facing description of the game:
    game_description = "Fast-paced pixel art platformer. Collect coins, avoid pits, and reach the flag before time runs out."

    # Should frames auto-advance or wait for user input?
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
        
        # --- Visuals & Colors ---
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_PLAYER = (255, 69, 0)  # Bright Red-Orange
        self.COLOR_PLATFORM = (139, 69, 19)  # Brown
        self.COLOR_COIN = (255, 215, 0)  # Gold
        self.COLOR_PIT_LAVA = (255, 100, 0) # Orange
        self.COLOR_PIT_LAVA_2 = (220, 20, 60) # Crimson
        self.COLOR_FLAG_POLE = (192, 192, 192) # Silver
        self.COLOR_FLAG = (0, 200, 0) # Green
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.font = pygame.font.SysFont('monospace', 24, bold=True)

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_STEPS = 1800 # 60 seconds * 30 fps
        self.GRAVITY = 0.8
        self.FRICTION = -0.12
        self.PLAYER_ACC = 0.9
        self.PLAYER_JUMP_STRENGTH = -15
        self.PLAYER_MAX_SPEED_X = 8

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.player_rect = None
        self.player_vel = None
        self.on_ground = False
        self.last_dist_to_flag = 0
        self.world_objects = {
            "platforms": [], "coins": [], "pits": [], "flag": None
        }
        self.camera_pos = [0, 0]
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.player_rect = pygame.Rect(100, 200, 20, 30)
        self.player_vel = [0, 0]
        self.on_ground = False
        
        self.particles = []

        self._generate_level()

        self.last_dist_to_flag = self._get_dist_to_flag()
        
        self.camera_pos = [self.player_rect.centerx - self.screen_width / 2, 
                           self.player_rect.centery - self.screen_height / 2]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        dist_before_move = self._get_dist_to_flag()

        self._update_player(movement)
        self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        self.time_remaining -= 1

        # --- Calculate Reward ---
        # 1. Reward for moving towards the flag
        dist_after_move = self._get_dist_to_flag()
        reward += dist_before_move - dist_after_move
        
        # 2. Penalty for safe/non-progressive actions
        if dist_after_move >= dist_before_move and movement == 0:
            reward -= 0.1 # Small penalty for standing still
        
        # Coin collection reward is handled in _handle_collisions
        reward_from_events = self._handle_collisions()
        reward += reward_from_events

        # --- Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.time_remaining <= 0
        
        if terminated and not self.game_over: # Time ran out
            reward -= 50
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.world_objects = {"platforms": [], "coins": [], "pits": [], "flag": None}
        
        # Start platform
        start_plat = pygame.Rect(0, 300, 300, 100)
        self.world_objects["platforms"].append(start_plat)
        
        last_plat = start_plat
        level_length = 40
        
        for i in range(level_length):
            gap = self.np_random.integers(50, 150)
            y_change = self.np_random.integers(-80, 80)
            width = self.np_random.integers(100, 400)
            
            new_x = last_plat.right + gap
            new_y = np.clip(last_plat.y + y_change, 150, self.screen_height - 20)
            
            new_plat = pygame.Rect(new_x, new_y, width, 200)
            self.world_objects["platforms"].append(new_plat)

            # Add coins
            if self.np_random.random() < 0.6:
                num_coins = self.np_random.integers(1, 4)
                for j in range(num_coins):
                    coin_x = new_plat.x + (new_plat.width / (num_coins + 1)) * (j + 1)
                    coin_y = new_plat.y - 40
                    self.world_objects["coins"].append(pygame.Rect(coin_x, coin_y, 15, 15))

            # Add pits based on difficulty scaling
            pit_frequency = min(0.6, 0.1 + 0.1 * (self.steps // 500))
            if self.np_random.random() < pit_frequency and i > 0:
                pit_x = last_plat.right
                pit_width = gap
                self.world_objects["pits"].append(pygame.Rect(pit_x, self.screen_height, pit_width, 100))

            last_plat = new_plat

        # Place flag on the last platform
        flag_pole = pygame.Rect(last_plat.centerx - 5, last_plat.top - 100, 5, 100)
        flag_banner = pygame.Rect(flag_pole.left - 30, flag_pole.top, 30, 20)
        self.world_objects["flag"] = {"pole": flag_pole, "banner": flag_banner}

    def _update_player(self, movement):
        # Horizontal movement
        acc = [0, self.GRAVITY]
        if movement == 3: # Left
            acc[0] = -self.PLAYER_ACC
        elif movement == 4: # Right
            acc[0] = self.PLAYER_ACC
        
        self.player_vel[0] += acc[0]
        self.player_vel[0] += self.player_vel[0] * self.FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED_X, self.PLAYER_MAX_SPEED_X)
        
        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump
            self._create_particles(self.player_rect.midbottom, 10, (139, 69, 19)) # Dust kick-up
        
        self.player_vel[1] += acc[1]

        # Update position
        self.player_rect.x += int(self.player_vel[0])
        self.player_rect.y += int(self.player_vel[1])

        self.on_ground = False # Assume not on ground until collision check proves otherwise

    def _handle_collisions(self):
        reward = 0
        
        # Platforms
        for plat in self.world_objects["platforms"]:
            if self.player_rect.colliderect(plat):
                # Check vertical collision
                if self.player_vel[1] > 0 and self.player_rect.bottom < plat.top + self.player_vel[1] + 1:
                    self.player_rect.bottom = plat.top
                    self.player_vel[1] = 0
                    if not self.on_ground: # Landing
                         # sfx: land
                         self._create_particles(self.player_rect.midbottom, 5, (139, 69, 19))
                    self.on_ground = True
                # Check horizontal collision
                elif self.player_vel[0] > 0 and self.player_rect.right < plat.left + self.player_vel[0] + 1:
                    self.player_rect.right = plat.left
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and self.player_rect.left > plat.right + self.player_vel[0] - 1:
                    self.player_rect.left = plat.right
                    self.player_vel[0] = 0
                # Check bottom collision (hitting head)
                elif self.player_vel[1] < 0:
                    self.player_rect.top = plat.bottom
                    self.player_vel[1] = 0
        
        # Coins
        collected_coins = []
        for coin in self.world_objects["coins"]:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                reward += 10
                self.score += 10
                # sfx: coin_collect
                self._create_particles(coin.center, 15, self.COLOR_COIN)
        self.world_objects["coins"] = [c for c in self.world_objects["coins"] if c not in collected_coins]

        # Pits
        for pit in self.world_objects["pits"]:
            if self.player_rect.colliderect(pit):
                self.game_over = True
                reward -= 100
                # sfx: fall_in_pit
                return reward
        
        # Fall off world
        if self.player_rect.top > self.screen_height + 200:
             self.game_over = True
             reward -= 100
             # sfx: fall_in_pit
             return reward

        # Flag
        if self.player_rect.colliderect(self.world_objects["flag"]["pole"]) or self.player_rect.colliderect(self.world_objects["flag"]["banner"]):
            self.game_over = True
            reward += 100
            self.score += 100
            # sfx: win
            self._create_particles(self.player_rect.center, 50, self.COLOR_FLAG)
            return reward

        return reward

    def _get_dist_to_flag(self):
        return abs(self.player_rect.centerx - self.world_objects["flag"]["pole"].centerx)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 1)],
                "life": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # Particle gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update camera with smooth interpolation (lerp)
        target_cam_x = self.player_rect.centerx - self.screen_width / 2
        target_cam_y = self.player_rect.centery - self.screen_height / 1.5
        self.camera_pos[0] += (target_cam_x - self.camera_pos[0]) * 0.1
        self.camera_pos[1] += (target_cam_y - self.camera_pos[1]) * 0.1

        cam_x, cam_y = int(self.camera_pos[0]), int(self.camera_pos[1])

        # Render Platforms
        for plat in self.world_objects["platforms"]:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, -cam_y))

        # Render Pits (with lava animation)
        for pit in self.world_objects["pits"]:
            pit_rect_on_screen = pit.move(-cam_x, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_PIT_LAVA_2, pit_rect_on_screen)
            lava_y_offset = math.sin(self.steps * 0.1) * 5
            top_lava_rect = pygame.Rect(pit_rect_on_screen.x, pit_rect_on_screen.y, pit_rect_on_screen.width, 10 + lava_y_offset)
            pygame.draw.rect(self.screen, self.COLOR_PIT_LAVA, top_lava_rect)

        # Render Coins (with spinning animation)
        for coin in self.world_objects["coins"]:
            w_offset = abs(math.sin(self.steps * 0.2 + coin.x) * (coin.width / 2))
            anim_rect = pygame.Rect(
                coin.x - cam_x + w_offset / 2, coin.y - cam_y,
                coin.width - w_offset, coin.height
            )
            pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)
            pygame.draw.ellipse(self.screen, (255, 255, 255), anim_rect, 1)

        # Render Flag
        flag = self.world_objects["flag"]
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, flag["pole"].move(-cam_x, -cam_y))
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag["banner"].move(-cam_x, -cam_y))
        
        # Render Particles
        for p in self.particles:
            pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
            pygame.draw.circle(self.screen, p["color"], pos, int(p["radius"] * (p["life"] / 20)))

        # Render Player
        player_screen_rect = self.player_rect.move(-cam_x, -cam_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_screen_rect)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score display
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, (10, 10), self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Time display
        time_left_sec = max(0, self.time_remaining // self.FPS)
        time_text = f"TIME: {time_left_sec}"
        time_text_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        time_pos = (self.screen_width - time_text_surf.get_width() - 10, 10)
        draw_text(time_text, time_pos, self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for visualization ---
    pygame.display.set_caption(env.game_description)
    window = pygame.display.set_mode((env.screen_width, env.screen_height))
    running = True
    
    total_reward = 0
    
    # --- Key mapping for human play ---
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement action (only one can be active)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1

        # Space and Shift are not used in this game
        # action[1] = 1 if keys[pygame.K_SPACE] else 0
        # action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Render to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()