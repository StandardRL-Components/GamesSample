
# Generated: 2025-08-28T02:41:01.646522
# Source Brief: brief_01773.md
# Brief Index: 1773

        
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
        "Controls: Use directional keys to jump. Up+Left/Right for long jumps, Down+Left/Right for short hops."
    )

    game_description = (
        "Leap between procedurally generated platforms, collecting coins and striving for the end flag in this fast-paced, side-scrolling arcade hopper."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.LEVEL_WIDTH = 6400
        
        # Colors
        self.COLOR_BG_TOP = (48, 25, 52)
        self.COLOR_BG_BOTTOM = (28, 15, 32)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_FLAG = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (10, 10, 10, 180)

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_VEL_V = 14
        self.JUMP_VEL_H = 7
        self.MAX_FALL_SPEED = 15

        # Fonts
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.is_on_ground = False
        self.platforms = []
        self.coins = []
        self.end_flag = None
        self.camera_x = 0.0
        self.platform_speed_multiplier = 1.0
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_speed_multiplier = 1.0
        
        self.player = {
            "rect": pygame.Rect(150, 200, 20, 20),
            "vx": 0.0,
            "vy": 0.0,
        }
        self.is_on_ground = False
        
        self._generate_level()
        self.camera_x = 0.0
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []

        start_plat = pygame.Rect(100, 350, 150, 20)
        self.platforms.append({"rect": start_plat, "vx": 0, "vy": 0, "range": 0, "initial_x": start_plat.x, "initial_y": start_plat.y})
        
        current_x = start_plat.right
        last_y = start_plat.top

        while current_x < self.LEVEL_WIDTH - 800:
            plat_width = self.np_random.integers(low=80, high=150)
            gap = self.np_random.integers(low=50, high=200)
            
            x = current_x + gap
            y = np.clip(last_y + self.np_random.integers(low=-100, high=100), 150, self.HEIGHT - 50)

            new_plat = pygame.Rect(x, y, plat_width, 20)
            
            vx, vy, move_range = 0, 0, 0
            if self.np_random.random() > 0.7:
                if self.np_random.random() > 0.5:
                    vx = self.np_random.choice([-1, 1])
                    move_range = self.np_random.integers(low=40, high=80)
                else:
                    vy = self.np_random.choice([-0.5, 0.5])
                    move_range = self.np_random.integers(low=30, high=60)

            self.platforms.append({
                "rect": new_plat, "vx": vx, "vy": vy, "range": move_range,
                "initial_x": new_plat.x, "initial_y": new_plat.y
            })

            if self.np_random.random() > 0.4:
                coin_rect = pygame.Rect(new_plat.centerx - 5, new_plat.top - 40, 10, 10)
                self.coins.append(coin_rect)

            current_x = new_plat.right
            last_y = new_plat.top

        flag_pole = pygame.Rect(current_x + 200, last_y - 100, 5, 120)
        flag_banner = pygame.Rect(flag_pole.right, flag_pole.top, 50, 30)
        self.end_flag = {"pole": flag_pole, "banner": flag_banner}
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement = action[0]
        
        if self.is_on_ground:
            jump_vx, jump_vy = 0.0, 0.0
            if movement == 1: # up-left
                jump_vx, jump_vy = -self.JUMP_VEL_H, -self.JUMP_VEL_V
            elif movement == 2: # up-right
                jump_vx, jump_vy = self.JUMP_VEL_H, -self.JUMP_VEL_V
            elif movement == 3: # down-left
                jump_vx, jump_vy = -self.JUMP_VEL_H / 1.5, -self.JUMP_VEL_V / 2
            elif movement == 4: # down-right
                jump_vx, jump_vy = self.JUMP_VEL_H / 1.5, -self.JUMP_VEL_V / 2
            
            if movement != 0:
                self.player["vx"] = jump_vx
                self.player["vy"] = jump_vy
                self.is_on_ground = False
                # sfx: jump sound

        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_speed_multiplier = min(2.5, self.platform_speed_multiplier + 0.05)

        for plat in self.platforms:
            if plat["vx"] != 0:
                plat["rect"].x += plat["vx"] * self.platform_speed_multiplier
                if abs(plat["rect"].x - plat["initial_x"]) > plat["range"]:
                    plat["vx"] *= -1
            if plat["vy"] != 0:
                plat["rect"].y += plat["vy"] * self.platform_speed_multiplier
                if abs(plat["rect"].y - plat["initial_y"]) > plat["range"]:
                    plat["vy"] *= -1

        if not self.is_on_ground:
            self.player["vy"] += self.GRAVITY
            self.player["vy"] = min(self.player["vy"], self.MAX_FALL_SPEED)
            reward -= 0.01
        else:
            self.player["vx"] = 0
            reward += 0.1

        prev_player_bottom = self.player["rect"].bottom
        self.player["rect"].x += self.player["vx"]
        self.player["rect"].y += self.player["vy"]

        self.is_on_ground = False
        player_rect = self.player["rect"]
        
        for plat in self.platforms:
            plat_rect = plat["rect"]
            if player_rect.colliderect(plat_rect):
                is_landing = (self.player["vy"] > 0) and (prev_player_bottom <= plat_rect.top)
                if is_landing:
                    player_rect.bottom = plat_rect.top
                    self.player["vy"] = 0
                    self.is_on_ground = True
                    player_rect.x += plat["vx"] * self.platform_speed_multiplier
                    # sfx: land sound
                    break
        
        collected_coins = []
        for coin in self.coins:
            if player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1
                # sfx: coin collect sound
        self.coins = [c for c in self.coins if c not in collected_coins]

        self.camera_x = self.player["rect"].centerx - self.WIDTH / 2
        self.camera_x = np.clip(self.camera_x, 0, self.LEVEL_WIDTH - self.WIDTH)
        
        terminated = False
        if player_rect.top > self.HEIGHT:
            terminated = True
            self.game_over = True
            reward -= 100
            # sfx: fall/death sound

        if player_rect.colliderect(self.end_flag["pole"]) or player_rect.colliderect(self.end_flag["banner"]):
            terminated = True
            self.game_over = True
            self.score += 100
            reward += 100
            # sfx: victory sound

        if self.steps >= 2000:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        for y in range(self.HEIGHT):
            r = np.interp(y, [0, self.HEIGHT], [self.COLOR_BG_TOP[0], self.COLOR_BG_BOTTOM[0]])
            g = np.interp(y, [0, self.HEIGHT], [self.COLOR_BG_TOP[1], self.COLOR_BG_BOTTOM[1]])
            b = np.interp(y, [0, self.HEIGHT], [self.COLOR_BG_TOP[2], self.COLOR_BG_BOTTOM[2]])
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))

        for plat in self.platforms:
            render_rect = plat["rect"].copy()
            render_rect.x -= self.camera_x
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect, border_radius=3)
        
        for coin in self.coins:
            render_rect = coin.copy()
            render_rect.x -= self.camera_x
            if render_rect.right > 0 and render_rect.left < self.WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, int(render_rect.centerx), int(render_rect.centery), int(coin.width / 2), self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, int(render_rect.centerx), int(render_rect.centery), int(coin.width / 2), self.COLOR_COIN)

        pole_rect = self.end_flag["pole"].copy()
        pole_rect.x -= self.camera_x
        banner_rect = self.end_flag["banner"].copy()
        banner_rect.x -= self.camera_x
        pygame.draw.rect(self.screen, (200,200,200), pole_rect)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, banner_rect)
        
        player_render_rect = self.player["rect"].copy()
        player_render_rect.x -= self.camera_x
        
        if not self.is_on_ground and self.player["vy"] < 0:
            px, py = player_render_rect.centerx, player_render_rect.bottom
            for _ in range(3):
                offset_x = self.np_random.uniform(-5, 5)
                offset_y = self.np_random.uniform(0, 10)
                size = self.np_random.uniform(1, 3)
                color = (
                    self.np_random.integers(200, 256),
                    self.np_random.integers(200, 256),
                    self.np_random.integers(100, 151)
                )
                pygame.draw.circle(self.screen, color, (px + offset_x, py + offset_y), size)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect, border_radius=2)
    
    def _render_ui(self):
        ui_surf = pygame.Surface((150, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (10, 10))

        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get an initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        
        # Test info from reset
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