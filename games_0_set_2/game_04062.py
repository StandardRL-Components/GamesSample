
# Generated: 2025-08-28T01:17:17.848820
# Source Brief: brief_04062.md
# Brief Index: 4062

        
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
    user_guide = (
        "Controls: Hold [Space] on a platform to charge a jump, release to leap. "
        "Use ← and → to control your movement in the air."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced geometric platformer. Charge your jumps and navigate "
        "procedurally generated levels to reach the green flag before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 255)
        self.COLOR_PLATFORM = (100, 110, 130)
        self.COLOR_PLATFORM_OUTLINE = (60, 70, 90)
        self.COLOR_FLAG = (0, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CHARGE_BAR_BG = (50, 50, 50)
        self.COLOR_CHARGE_BAR_FG = (255, 255, 0)
        
        # Physics & Gameplay
        self.GRAVITY = 0.7
        self.PLAYER_SIZE = 20
        self.AIR_CONTROL_FORCE = 0.6
        self.MAX_AIR_SPEED = 6.0
        self.FRICTION = 0.95
        self.JUMP_CHARGE_RATE = 0.04
        self.MAX_JUMP_CHARGE = 1.0
        self.MIN_JUMP_POWER = 4.0
        self.MAX_JUMP_POWER = 16.0
        self.NUM_PLATFORMS = 25
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False
        self.is_charging = False
        self.jump_charge = 0.0
        self.space_was_held = False
        self.platforms = []
        self.flag_pos = (0, 0)
        self.camera_y = 0.0
        self.particles = []
        self.highest_platform_reached = 0

        # Initialize state for the first time
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.math.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE)
        self.player_vel = pygame.math.Vector2(0, 0)
        
        self.on_ground = True
        self.is_charging = False
        self.jump_charge = 0.0
        self.space_was_held = False
        self.particles = []
        
        self.camera_y = self.player_pos.y - self.HEIGHT * 0.75
        self.highest_platform_reached = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01 # Small penalty per step to encourage speed
        
        self._handle_input(movement, space_held)
        self._update_physics()
        landing_reward = self._handle_collisions()
        reward += landing_reward
        self._update_camera()
        self._update_particles()
        
        self.timer -= 1
        self.steps += 1
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Air Control ---
        if not self.on_ground:
            if movement == 3: # Left
                self.player_vel.x -= self.AIR_CONTROL_FORCE
            elif movement == 4: # Right
                self.player_vel.x += self.AIR_CONTROL_FORCE
            self.player_vel.x = max(-self.MAX_AIR_SPEED, min(self.MAX_AIR_SPEED, self.player_vel.x))

        # --- Jump Charging ---
        if space_held and self.on_ground:
            self.is_charging = True
            self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            # sfx: charging_sound_loop
        
        # --- Jump Execution (on release) ---
        if not space_held and self.space_was_held and self.on_ground:
            if self.is_charging:
                jump_power = self.MIN_JUMP_POWER + (self.MAX_JUMP_POWER - self.MIN_JUMP_POWER) * self.jump_charge
                self.player_vel.y = -jump_power
                self.on_ground = False
                self.is_charging = False
                self._create_particles(self.player_pos + (self.PLAYER_SIZE/2, self.PLAYER_SIZE), 15, self.COLOR_PLAYER) # Jump burst
                # sfx: jump_release
            self.jump_charge = 0.0

        self.space_was_held = space_held

    def _update_physics(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        self.player_vel.x *= self.FRICTION
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        reward = 0.0

        if self.player_vel.y > 0: # Only check for landing if moving down
            for i, platform in enumerate(self.platforms):
                if player_rect.colliderect(platform):
                    # Check if player was above the platform in the previous frame
                    if self.player_pos.y + self.PLAYER_SIZE - self.player_vel.y <= platform.top:
                        self.player_pos.y = platform.top - self.PLAYER_SIZE
                        self.player_vel.y = 0
                        self.player_vel.x = 0 # Stop horizontal movement on landing
                        self.on_ground = True
                        
                        if i > self.highest_platform_reached:
                            reward += 1.0 * (i - self.highest_platform_reached) # Reward for reaching a new, higher platform
                            self.highest_platform_reached = i
                        
                        self._create_particles(player_rect.midbottom, 8, self.COLOR_PLATFORM) # Landing dust
                        # sfx: land_thud
                        break
        return reward

    def _check_termination(self):
        # Win condition: reaching the flag
        flag_rect = pygame.Rect(self.flag_pos[0] - 10, self.flag_pos[1] - 20, 20, 20)
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if player_rect.colliderect(flag_rect):
            # sfx: victory_fanfare
            return True, 100.0

        # Lose condition: falling off screen
        if self.player_pos.y > self.camera_y + self.HEIGHT + self.PLAYER_SIZE * 2:
            # sfx: falling_whistle
            return True, -5.0

        # Lose condition: time out or max steps
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            # sfx: timeout_buzzer
            return True, -10.0

        return False, 0.0

    def _update_camera(self):
        # Smoothly follow the player, keeping them in the upper half of the screen
        target_camera_y = self.player_pos.y - self.HEIGHT * 0.4
        self.camera_y += (target_camera_y - self.camera_y) * 0.08

    def _generate_platforms(self):
        self.platforms = []
        start_platform = pygame.Rect(self.WIDTH / 2 - 60, self.HEIGHT - 50, 120, 20)
        self.platforms.append(start_platform)
        
        last_platform = start_platform
        for _ in range(self.NUM_PLATFORMS):
            y_offset = self.np_random.integers(low=70, high=130)
            x_offset = self.np_random.integers(low=-180, high=180)
            width = self.np_random.integers(low=70, high=140)
            
            new_x = last_platform.centerx + x_offset
            new_x = max(width / 2, min(self.WIDTH - width / 2, new_x))
            
            new_platform = pygame.Rect(new_x - width / 2, last_platform.y - y_offset, width, 20)
            self.platforms.append(new_platform)
            last_platform = new_platform
            
        self.flag_pos = (last_platform.centerx, last_platform.top)

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        for p in self.particles:
            pos, vel, life, color = p
            pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y - self.camera_y)), int(life))

        for p in self.platforms:
            screen_rect = p.move(0, -self.camera_y)
            if screen_rect.bottom > 0 and screen_rect.top < self.HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, screen_rect, width=2, border_radius=3)

        flag_screen_pos = (int(self.flag_pos[0]), int(self.flag_pos[1] - self.camera_y))
        p1 = (flag_screen_pos[0], flag_screen_pos[1] - 20)
        p2 = (flag_screen_pos[0], flag_screen_pos[1])
        p3 = (flag_screen_pos[0] + 15, flag_screen_pos[1] - 10)
        pygame.draw.line(self.screen, self.COLOR_FLAG, flag_screen_pos, p1, 2)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p3, p2], self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p3, p2], self.COLOR_FLAG)

        player_screen_pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
        player_rect = pygame.Rect(player_screen_pos, (self.PLAYER_SIZE, self.PLAYER_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=3)

        if self.is_charging:
            bar_width, bar_height = 40, 8
            bg_rect = pygame.Rect(player_rect.centerx - bar_width / 2, player_rect.top - 20, bar_width, bar_height)
            fg_width = bar_width * self.jump_charge
            fg_rect = pygame.Rect(bg_rect.left, bg_rect.top, fg_width, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_BG, bg_rect, border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_FG, fg_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_str = f"TIME: {self.timer / self.FPS:.1f}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        if self.game_over:
            flag_rect = pygame.Rect(self.flag_pos[0] - 10, self.flag_pos[1] - 20, 20, 20)
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
            is_win = player_rect.colliderect(flag_rect)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "SUCCESS!" if is_win else "GAME OVER"
            end_color = self.COLOR_FLAG if is_win else (255, 80, 80)
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": round(self.timer / self.FPS, 2),
            "player_y": self.player_pos.y
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.uniform(3, 8)
            self.particles.append([pygame.math.Vector2(pos), vel, life, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 0.2
        self.particles = [p for p in self.particles if p[2] > 0]

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    import time

    env = GameEnv()
    env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Jumper Game - Human Play")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    print(env.user_guide)
    print("Press 'R' to reset the environment.")

    while running:
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_reward = 0.0
                env.reset()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
            time.sleep(2)
            total_reward = 0.0
            env.reset()
            
        clock.tick(env.FPS)
        
    env.close()