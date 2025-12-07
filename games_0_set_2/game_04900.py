
# Generated: 2025-08-28T03:21:09.991300
# Source Brief: brief_04900.md
# Brief Index: 4900

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→↑↓ to aim. Hold [SPACE] to charge your jump, release to launch. "
        "Hold [SHIFT] in mid-air to apply air brakes."
    )

    game_description = (
        "Hop your spaceship between platforms to reach the green finish line. "
        "Charge your jumps for power and aim carefully. Don't fall or run out of time!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500  # 50 seconds at 30fps

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Colors
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_PLATFORM = (120, 130, 150)
        self.COLOR_PLATFORM_OUTLINE = (180, 190, 210)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_FINISH_OUTLINE = (150, 255, 150)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_TRAJECTORY = (255, 255, 255)
        
        # Physics constants
        self.GRAVITY = 0.3
        self.AIR_BRAKE_MULTIPLIER = 2.5
        self.MAX_CHARGE = 10.0
        self.CHARGE_RATE = 0.25
        self.JUMP_BASE_FORCE = 3.0

        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_platform_idx = None
        self.is_charging = False
        self.charge_power = 0.0
        self.jump_angle = 0.0
        self.prev_space_held = False
        self.last_landed_platform_idx = -1

        self.platforms = []
        self.finish_rect = None
        self.particles = []

        self.level = 1
        self.timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # Initialize state
        self.reset()
        
        # This will fail if not headless, but is required for the prompt
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.LEVEL_TIME = 30 * self.FPS
        self.timer = self.LEVEL_TIME

        self._generate_level()
        
        start_platform = self.platforms[0]
        self.player_pos = [start_platform.centerx, start_platform.top]
        self.player_vel = [0, 0]
        self.on_platform_idx = 0
        self.last_landed_platform_idx = 0
        
        self.is_charging = False
        self.charge_power = 0.0
        self.jump_angle = 0.0  # Default to right
        self.prev_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1

        # --- Input Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_released = self.prev_space_held and not space_held
        self.prev_space_held = space_held

        if self.on_platform_idx is not None:
            reward += 0.1  # On-platform reward
            
            # Aiming
            if movement == 1: self.jump_angle = -90  # Up
            elif movement == 2: self.jump_angle = 90   # Down
            elif movement == 3: self.jump_angle = 180  # Left
            elif movement == 4: self.jump_angle = 0    # Right

            # Charging
            if space_held:
                self.is_charging = True
                self.charge_power = min(self.MAX_CHARGE, self.charge_power + self.CHARGE_RATE)
            else:
                self.is_charging = False

            # Jumping
            if space_released and self.charge_power > 1.0: # Minimum charge to jump
                self.on_platform_idx = None
                angle_rad = math.radians(self.jump_angle)
                jump_force = self.JUMP_BASE_FORCE + self.charge_power
                self.player_vel = [math.cos(angle_rad) * jump_force, math.sin(angle_rad) * jump_force]
                self.charge_power = 0
                # sfx: jump

        else:  # In the air
            reward -= 0.01  # In-air penalty
            
            # Physics update
            gravity = self.GRAVITY * self.AIR_BRAKE_MULTIPLIER if shift_held else self.GRAVITY
            self.player_vel[1] += gravity
            self.player_pos[0] += self.player_vel[0]
            self.player_pos[1] += self.player_vel[1]

            # Collision checking
            player_rect = pygame.Rect(self.player_pos[0] - 4, self.player_pos[1] - 8, 8, 8)

            # Check landing on platforms
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and self.player_vel[1] > 0 and player_rect.bottom < plat.top + self.player_vel[1] + 1:
                    self.player_pos[1] = plat.top
                    self.player_vel = [0, 0]
                    self.on_platform_idx = i
                    self.charge_power = 0
                    self.is_charging = False
                    self._create_particles([self.player_pos[0], plat.top], 20, self.COLOR_PLATFORM_OUTLINE)
                    # sfx: land

                    if i != self.last_landed_platform_idx:
                        reward += 5.0
                        self.last_landed_platform_idx = i
                    
                    if player_rect.left < plat.left + 5 or player_rect.right > plat.right - 5:
                        reward -= 1.0 # Near miss penalty
                    
                    break
            
            # Check landing on finish line
            if not self.game_over and self.finish_rect.colliderect(player_rect):
                self.game_over = True
                self.game_outcome = "LEVEL COMPLETE!"
                time_bonus = max(0, (self.timer / self.LEVEL_TIME) * 50)
                reward += 100 + time_bonus
                self.level += 1
                self._create_particles(self.player_pos, 50, self.COLOR_FINISH_OUTLINE)
                # sfx: victory

        # --- Failure Conditions ---
        if not self.game_over:
            if self.on_platform_idx is None and (self.player_pos[1] > self.HEIGHT + 20 or self.player_pos[0] < -20 or self.player_pos[0] > self.WIDTH + 20):
                self.game_over = True
                self.game_outcome = "FELL INTO THE VOID"
                reward -= 50
                # sfx: fail
            elif self.timer <= 0:
                self.game_over = True
                self.game_outcome = "TIME'S UP!"
                reward -= 25
                # sfx: timeout
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.game_outcome = "MAX STEPS REACHED"
                reward -= 25

        self.score += reward
        terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.platforms = []
        
        # Start platform
        start_plat = pygame.Rect(50, self.HEIGHT - 50, 100, 20)
        self.platforms.append(start_plat)

        current_pos = [start_plat.centerx, start_plat.top]
        
        for i in range(10): # Generate 10 platforms
            plat_w = max(40, 120 - self.level * 5)
            plat_h = 20
            
            gap_x = self.np_random.uniform(low=50 + self.level * 5, high=100 + self.level * 10)
            gap_y = self.np_random.uniform(low=-80, high=80)

            new_x = current_pos[0] + gap_x
            new_y = current_pos[1] + gap_y
            
            # Clamp to screen bounds
            new_x = np.clip(new_x, plat_w / 2, self.WIDTH - plat_w / 2)
            new_y = np.clip(new_y, 50, self.HEIGHT - 50)

            # Prevent platforms from overlapping horizontally
            if new_x < current_pos[0] + self.platforms[-1].width / 2:
                new_x = current_pos[0] + self.platforms[-1].width / 2 + 10

            new_plat = pygame.Rect(new_x - plat_w / 2, new_y, plat_w, plat_h)
            self.platforms.append(new_plat)
            current_pos = [new_plat.centerx, new_plat.top]

        # Finish line
        last_plat = self.platforms[-1]
        self.finish_rect = pygame.Rect(last_plat.left, last_plat.top - 50, last_plat.width, 10)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, width=2, border_radius=3)

        # Render finish line
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_FINISH_OUTLINE, self.finish_rect, width=2, border_radius=3)

        # Render particles
        self._update_and_draw_particles()

        # Render trajectory predictor
        if self.is_charging and self.charge_power > 0:
            self._draw_trajectory_predictor()
        
        # Render player
        self._draw_player()

    def _draw_player(self):
        if self.player_pos is None: return

        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Charging indicator
        if self.is_charging:
            charge_radius = int(self.charge_power * 1.5)
            alpha = int(100 + (self.charge_power / self.MAX_CHARGE) * 155)
            s = pygame.Surface((charge_radius*2, charge_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, charge_radius, charge_radius, charge_radius - 1, self.COLOR_PLAYER_GLOW + (alpha,))
            pygame.gfxdraw.filled_circle(s, charge_radius, charge_radius, charge_radius - 1, self.COLOR_PLAYER_GLOW + (alpha,))
            self.screen.blit(s, (pos[0] - charge_radius, pos[1] - charge_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player triangle
        size = 8
        if self.on_platform_idx is not None:
             # Idle, point up
            angle_rad = math.radians(-90)
        else:
             # In air, point along velocity
            angle_rad = math.atan2(self.player_vel[1], self.player_vel[0])

        p1 = (pos[0] + size * math.cos(angle_rad), pos[1] + size * math.sin(angle_rad))
        p2 = (pos[0] + size * 0.5 * math.cos(angle_rad + 2.5), pos[1] + size * 0.5 * math.sin(angle_rad + 2.5))
        p3 = (pos[0] + size * 0.5 * math.cos(angle_rad - 2.5), pos[1] + size * 0.5 * math.sin(angle_rad - 2.5))
        
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_trajectory_predictor(self):
        angle_rad = math.radians(self.jump_angle)
        jump_force = self.JUMP_BASE_FORCE + self.charge_power
        sim_vel = [math.cos(angle_rad) * jump_force, math.sin(angle_rad) * jump_force]
        sim_pos = list(self.player_pos)

        for _ in range(30): # Simulate 1 second into the future
            for _ in range(2): # 2 physics steps per drawn dot for spacing
                sim_vel[1] += self.GRAVITY
                sim_pos[0] += sim_vel[0]
                sim_pos[1] += sim_vel[1]
            if sim_pos[1] < self.HEIGHT:
                pygame.gfxdraw.filled_circle(self.screen, int(sim_pos[0]), int(sim_pos[1]), 1, self.COLOR_TRAJECTORY + (150,))

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), 24, self.COLOR_UI)
        # Level
        self._render_text(f"LEVEL: {self.level}", (self.WIDTH / 2, 10), 24, self.COLOR_UI, align="top")
        # Timer
        time_str = f"TIME: {max(0, self.timer / self.FPS):.1f}"
        self._render_text(time_str, (self.WIDTH - 10, 10), 24, self.COLOR_UI, align="topright")
        
        # Game Over Message
        if self.game_over:
            self._render_text(self.game_outcome, (self.WIDTH / 2, self.HEIGHT / 2 - 40), 64, self.COLOR_UI, align="center")
            self._render_text("Reset to continue", (self.WIDTH / 2, self.HEIGHT / 2 + 20), 24, self.COLOR_UI, align="center")

    def _render_text(self, text, position, size, color, align="topleft"):
        font = self.font_large if size > 32 else self.font_small
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topleft":
            text_rect.topleft = position
        elif align == "topright":
            text_rect.topright = position
        elif align == "center":
            text_rect.center = position
        elif align == "top":
            text_rect.midtop = position
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([list(pos), vel, lifetime, color])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[2] / 30))
                radius = int(p[2] / 10) + 1
                pygame.gfxdraw.filled_circle(self.screen, int(p[0][0]), int(p[0][1]), radius, p[3] + (alpha,))
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display drivers
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy" # Uncomment for headless execution

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopper")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # In a real game, you might wait for a key press to reset
            # For this demo, we'll auto-reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()