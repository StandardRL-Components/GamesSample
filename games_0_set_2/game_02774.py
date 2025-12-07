
# Generated: 2025-08-27T21:24:25.207599
# Source Brief: brief_02774.md
# Brief Index: 2774

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Hold [SPACE] to charge your jump, release to leap. Reach the top platform before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade platformer. Hop upwards between procedurally generated platforms, "
        "managing your jump power to reach the top. The world constantly scrolls up, so don't fall behind!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS # 60 seconds

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150, 100)
        self.COLOR_PLATFORM = (220, 220, 220)
        self.COLOR_GOAL_PLATFORM = (0, 255, 128)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_CHARGE_BAR = (0, 255, 128)
        self.COLOR_CHARGE_BAR_BG = (100, 100, 100)

        # Physics & Gameplay
        self.GRAVITY = 0.35
        self.SCROLL_SPEED = 1.0
        self.JUMP_CHARGE_RATE = 0.5
        self.MAX_JUMP_CHARGE = 15
        self.PLAYER_RADIUS = 10
        self.NUM_PLATFORMS = 12
        self.GOAL_PLATFORM_Y = -self.HEIGHT * 4 # Far above start
        self.PLATFORM_HEIGHT = 10

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.jump_charge = None
        self.was_charging = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.highest_platform_y = None
        self.platform_width_variance = None
        self.platform_gap_variance = None

        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.jump_charge = 0
        self.was_charging = False
        self.particles = deque()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.platform_width_variance = 0.2
        self.platform_gap_variance = 10

        self._generate_platforms()
        self.highest_platform_y = self.platforms[0].y

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms = deque()
        
        start_plat = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 40, 100, self.PLATFORM_HEIGHT)
        self.platforms.append(start_plat)
        
        last_y = start_plat.y
        base_width = 90
        base_gap = 70

        for i in range(self.NUM_PLATFORMS - 1):
            width = base_width * (1 - self.platform_width_variance + self.np_random.random() * 2 * self.platform_width_variance)
            gap = base_gap + self.np_random.uniform(-self.platform_gap_variance, self.platform_gap_variance)
            
            x = self.np_random.uniform(self.PLAYER_RADIUS, self.WIDTH - width - self.PLAYER_RADIUS)
            y = last_y - gap - self.PLATFORM_HEIGHT
            
            self.platforms.append(pygame.Rect(x, y, width, self.PLATFORM_HEIGHT))
            last_y = y
            
        self.goal_platform = pygame.Rect(0, self.GOAL_PLATFORM_Y, self.WIDTH, self.PLATFORM_HEIGHT * 2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(space_held)
        self._update_physics()
        self._update_world()

        landing_reward, new_highest = self._check_collisions()
        reward += landing_reward
        if new_highest:
            reward += 5.0 # Reached new highest platform

        self.steps += 1
        
        if self.steps > 0:
            if self.steps % 500 == 0:
                self.platform_width_variance = min(0.5, self.platform_width_variance + 0.1)
            if self.steps % 250 == 0:
                self.platform_gap_variance = min(35, self.platform_gap_variance + 1)

        terminated = self._check_termination()
        if terminated:
            if self.player_pos.y > self.HEIGHT + self.PLAYER_RADIUS:
                reward -= 10.0
            elif self.steps >= self.MAX_STEPS:
                reward -= 10.0
            elif self.player_pos.y <= self.goal_platform.bottom:
                reward += 100.0
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, space_held):
        if space_held and self.on_ground:
            self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            self.was_charging = True
        elif not space_held and self.was_charging:
            self.player_vel.y = -self.jump_charge
            self.on_ground = False
            self.jump_charge = 0
            self.was_charging = False
            # sfx: jump

    def _update_physics(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

    def _update_world(self):
        self.player_pos.y += self.SCROLL_SPEED
        for plat in self.platforms:
            plat.y += self.SCROLL_SPEED
        self.goal_platform.y += self.SCROLL_SPEED
        self.highest_platform_y += self.SCROLL_SPEED

        if self.platforms and self.platforms[0].top > self.HEIGHT:
            self.platforms.popleft()
            last_plat = self.platforms[-1]
            base_width = 90
            base_gap = 70
            width = base_width * (1 - self.platform_width_variance + self.np_random.random() * 2 * self.platform_width_variance)
            gap = base_gap + self.np_random.uniform(-self.platform_gap_variance, self.platform_gap_variance)
            x = self.np_random.uniform(self.PLAYER_RADIUS, self.WIDTH - width - self.PLAYER_RADIUS)
            y = last_plat.y - gap - self.PLATFORM_HEIGHT
            self.platforms.append(pygame.Rect(x, y, width, self.PLATFORM_HEIGHT))
            
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        landing_reward = 0.0
        new_highest = False
        
        if self.player_vel.y > 0:
            all_platforms = list(self.platforms) + [self.goal_platform]
            for plat in all_platforms:
                if (plat.left < self.player_pos.x < plat.right and
                    plat.top <= self.player_pos.y + self.PLAYER_RADIUS and
                    plat.top > self.player_pos.y + self.PLAYER_RADIUS - self.player_vel.y - self.SCROLL_SPEED):
                    
                    self.on_ground = True
                    self.player_vel.y = 0
                    self.player_pos.y = plat.top - self.PLAYER_RADIUS
                    landing_reward = 1.0
                    # sfx: land
                    
                    if plat.y < self.highest_platform_y:
                        self.highest_platform_y = plat.y
                        new_highest = True

                    for _ in range(10):
                        angle = self.np_random.uniform(0, math.pi)
                        speed = self.np_random.uniform(1, 3)
                        vel = pygame.Vector2(math.cos(angle) * speed, -math.sin(angle) * speed)
                        self.particles.append({
                            'pos': pygame.Vector2(self.player_pos.x, self.player_pos.y + self.PLAYER_RADIUS),
                            'vel': vel,
                            'life': self.np_random.integers(10, 20)
                        })
                    break
        return landing_reward, new_highest

    def _check_termination(self):
        if self.player_pos.y > self.HEIGHT + self.PLAYER_RADIUS: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.player_pos.y <= self.goal_platform.bottom: return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "height": -self.player_pos.y
        }

    def _get_observation(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
        
        pygame.draw.rect(self.screen, self.COLOR_GOAL_PLATFORM, self.goal_platform, border_radius=5)
        
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), max(0, int(p['life']/5)), self.COLOR_PARTICLE)

        player_x, player_y = int(self.player_pos.x), int(self.player_pos.y)
        
        glow_surface = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS * 1.5)
        self.screen.blit(glow_surface, (player_x - self.PLAYER_RADIUS * 2, player_y - self.PLAYER_RADIUS * 2))

        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        height_val = max(0, int((self.HEIGHT - 40 - self.player_pos.y) / 10))
        height_text = self.font_main.render(f"HEIGHT: {height_val}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        if self.on_ground and self.jump_charge > 0:
            bar_width, bar_height = 50, 8
            bar_x, bar_y = self.player_pos.x - bar_width / 2, self.player_pos.y - self.PLAYER_RADIUS - 20
            fill_ratio = self.jump_charge / self.MAX_JUMP_CHARGE
            fill_width = bar_width * fill_ratio
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=2)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Jumper")
    clock = pygame.time.Clock()
    
    running = True
    action = env.action_space.sample()
    action.fill(0)

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        action[0] = 0
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()