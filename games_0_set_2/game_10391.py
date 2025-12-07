import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:24:16.522854
# Source Brief: brief_00391.md
# Brief Index: 391
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Master momentum and timing to chain jumps and launch projectiles for maximum points
    in a fast-paced, single-button arcade experience.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (Ignored)
    - actions[1]: Space button (0=released, 1=held) - Charges and releases jumps/projectiles.
    - actions[2]: Shift button (Ignored)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Master momentum and timing to chain jumps and launch projectiles at targets for maximum points."
    user_guide = "Hold [space] to charge your jump. Release to launch yourself and fire a projectile."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TARGET_FPS = 60
    MAX_EPISODE_STEPS = 90 * TARGET_FPS  # 90 seconds
    WIN_SCORE = 100

    # Colors
    COLOR_BG = (5, 0, 15)
    COLOR_GROUND = (0, 119, 255)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE = (255, 51, 51)
    COLOR_TARGET = (255, 215, 0)
    COLOR_METER = (0, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)

    # Physics & Gameplay
    GRAVITY = 0.5
    GROUND_Y = HEIGHT - 50
    PLAYER_X = 100
    PLAYER_SIZE = 20
    
    CHARGE_RATE = 0.02
    MAX_CHARGE = 1.0
    
    JUMP_VEL_MULTIPLIER = -20.0
    PROJ_VEL_MULTIPLIER = 15.0
    
    TARGET_SIZE = 40
    TARGET_Y = GROUND_Y - 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel_y = 0.0
        self.player_state = "grounded"  # grounded, charging, jumping
        self.charge_level = 0.0
        self.last_space_held = False
        self.projectiles = []
        self.particles = []
        self.target_pos = pygame.Vector2(0, 0)
        self.last_player_y = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        self.player_pos = pygame.Vector2(self.PLAYER_X, self.GROUND_Y)
        self.player_vel_y = 0.0
        self.player_state = "grounded"
        self.charge_level = 0.0
        self.last_space_held = False
        
        self.projectiles.clear()
        self.particles.clear()
        
        self._spawn_target()
        self.last_player_y = self.player_pos.y

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0.0

        # 1. Handle Input
        self._handle_input(space_held)
        
        # 2. Update Game State
        prev_player_y = self.player_pos.y
        self._update_player()
        self._update_projectiles()
        self._update_particles()
        
        # 3. Collision Detection & Scoring
        reward += self._check_collisions()
        
        # 4. Continuous Reward for jumping up
        height_gained = prev_player_y - self.player_pos.y
        if height_gained > 0:
            reward += 0.1 * height_gained

        # 5. Check Termination
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward = 100.0  # Goal-oriented win reward
        elif self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            reward = -100.0  # Goal-oriented lose reward

        self.last_space_held = space_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, space_held):
        is_release = self.last_space_held and not space_held
        
        if self.player_state == "grounded" and space_held:
            self.player_state = "charging"
            self.charge_level = 0.0

        elif self.player_state == "charging":
            if space_held:
                self.charge_level = min(self.MAX_CHARGE, self.charge_level + self.CHARGE_RATE)
            if is_release:
                self.player_state = "jumping"
                self.player_vel_y = self.charge_level * self.JUMP_VEL_MULTIPLIER
                self._spawn_projectile()
                self.charge_level = 0.0

    def _update_player(self):
        if self.player_state == "jumping":
            self.player_vel_y += self.GRAVITY
            self.player_pos.y += self.player_vel_y
            
            if self.player_pos.y >= self.GROUND_Y:
                self.player_pos.y = self.GROUND_Y
                self.player_vel_y = 0.0
                self.player_state = "grounded"
                self._spawn_explosion(self.player_pos, 10, self.COLOR_GROUND, 1.5)

    def _update_projectiles(self):
        for proj in self.projectiles:
            proj['vel'].y += self.GRAVITY / 2 # Lighter gravity for projectiles
            proj['pos'] += proj['vel']
            proj['trail'].append(proj['pos'].copy())
            if len(proj['trail']) > 15:
                proj['trail'].pop(0)
        
        # Cleanup off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'].x < self.WIDTH and p['pos'].y < self.HEIGHT + 50]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_collisions(self):
        reward = 0
        target_rect = pygame.Rect(self.target_pos.x - self.TARGET_SIZE / 2, 
                                  self.target_pos.y - self.TARGET_SIZE / 2, 
                                  self.TARGET_SIZE, self.TARGET_SIZE)

        for proj in self.projectiles[:]:
            if target_rect.collidepoint(proj['pos']):
                reward += 15.0
                self.score += 15
                self.projectiles.remove(proj)
                self._spawn_explosion(self.target_pos, 50, self.COLOR_TARGET, 3.0)
                self._spawn_target()
                break
        return reward

    def _spawn_projectile(self):
        launch_angle = math.radians(45)
        launch_speed = 2 + self.charge_level * self.PROJ_VEL_MULTIPLIER
        
        vel = pygame.Vector2(
            launch_speed * math.cos(launch_angle),
            -launch_speed * math.sin(launch_angle) # Y is inverted in pygame
        )
        
        self.projectiles.append({
            'pos': self.player_pos.copy(),
            'vel': vel,
            'trail': []
        })

    def _spawn_target(self):
        # Spawn target on the right half of the screen
        x = self.np_random.uniform(self.WIDTH / 2, self.WIDTH - 50)
        self.target_pos = pygame.Vector2(x, self.TARGET_Y)

    def _spawn_explosion(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        pygame.draw.line(self.screen, tuple(min(255, c + 50) for c in self.COLOR_GROUND), (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)

        # Target
        self._draw_glowing_rect(self.screen, self.COLOR_TARGET, self.target_pos, self.TARGET_SIZE, 10, 80)
        
        # Projectiles
        for proj in self.projectiles:
            # Trail
            for i, p in enumerate(proj['trail']):
                alpha = int(255 * (i / len(proj['trail'])))
                color = (*self.COLOR_PROJECTILE, alpha)
                temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (3, 3), 3)
                self.screen.blit(temp_surf, (int(p.x - 3), int(p.y - 3)))
            # Main projectile
            self._draw_glowing_circle(self.screen, self.COLOR_PROJECTILE, proj['pos'], 6, 8, 100)

        # Player
        self._draw_glowing_rect(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_SIZE, 15, 60)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))
            
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:03d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left_sec = (self.MAX_EPISODE_STEPS - self.steps) / self.TARGET_FPS
        time_color = self.COLOR_TIMER_WARN if time_left_sec < 10 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"{time_left_sec:04.1f}", True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 20, 10))

        # Charge Meter
        if self.player_state == "charging":
            meter_height = 100
            meter_width = 15
            meter_x = 20
            meter_y = self.HEIGHT - 40 - meter_height
            
            fill_height = self.charge_level * meter_height
            
            # Draw background
            pygame.draw.rect(self.screen, (50, 50, 50), (meter_x, meter_y, meter_width, meter_height), 0, 4)
            # Draw fill
            if fill_height > 0:
                fill_color = self.COLOR_METER
                if self.charge_level >= 1.0:
                    # Flash when full
                    fill_color = (255, 255, 0) if (self.steps // 3) % 2 == 0 else self.COLOR_METER
                pygame.draw.rect(self.screen, fill_color, (meter_x, meter_y + meter_height - fill_height, meter_width, fill_height), 0, 4)
            # Draw border
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (meter_x, meter_y, meter_width, meter_height), 2, 4)

    def _draw_glowing_rect(self, surface, color, center_pos, size, glow_size, alpha):
        rect = pygame.Rect(center_pos.x - size/2, center_pos.y - size/2, size, size)
        for i in range(glow_size, 0, -2):
            glow_color = (*color, int(alpha * ((glow_size - i) / glow_size)))
            glow_surf = pygame.Surface((size + i*2, size + i*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=int(size/4 + i))
            surface.blit(glow_surf, (rect.x - i, rect.y - i))
        pygame.draw.rect(surface, color, rect, border_radius=int(size/4))

    def _draw_glowing_circle(self, surface, color, center, radius, glow_size, alpha):
        center_int = (int(center.x), int(center.y))
        for i in range(glow_size, 0, -1):
            current_alpha = int(alpha * ((glow_size - i) / glow_size))
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius + i, (*color, current_alpha))
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.font.quit()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # Manual play loop
    # Un-comment the next line to run with display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Jump & Launch")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        # Convert to MultiDiscrete action format
        action = [0, 1 if space_held else 0, 0] # Movement and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc

        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()