import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to run. Press ↑ or SPACE to jump. Hold SHIFT while jumping for a long jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced platformer. Guide your robot across scrolling platforms to reach the finish line before time runs out! Collect gold stars for a higher score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.COLOR_BG_TOP = (255, 180, 120)
        self.COLOR_BG_BOTTOM = (255, 230, 150)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_PLATFORM_EDGE = (150, 150, 170)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255, 100)
        self.COLOR_BONUS = (255, 215, 0)
        self.COLOR_FINISH = (0, 200, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRAVITY = 0.5
        self.MAX_STEPS = 30 * 30  # 30 seconds at 30fps
        self.FINISH_LINE_X = 5000
        self.PLAYER_WIDTH = 24
        self.PLAYER_HEIGHT = 36
        self.PLAYER_RUN_SPEED = 4
        self.PLAYER_JUMP_POWER = -10
        self.PLAYER_LONG_JUMP_POWER = -13

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.timer = 0
        self.world_scroll_x = 0
        self.scroll_speed = 0
        self.min_platform_gap = 0
        self.max_platform_gap = 0
        self.player_pos = [0.0, 0.0]
        self.player_vel = [0.0, 0.0]
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.player_on_ground = False
        self.player_anim_timer = 0
        self.particles = []
        self.platforms = []
        self.bonuses = []
        self.finish_flag_rect = pygame.Rect(0, 0, 0, 0)
        self.last_platform_landed = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.timer = self.MAX_STEPS
        
        self.world_scroll_x = 0
        self.scroll_speed = 3.0
        self.min_platform_gap = 50
        self.max_platform_gap = 150
        
        self.player_pos = [100.0, 200.0]
        self.player_vel = [0.0, 0.0]
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.player_on_ground = False
        self.player_anim_timer = 0
        
        self.particles = []
        self.platforms = []
        
        p_x = -50
        for i in range(15):
            p_w = self.np_random.integers(100, 300)
            p_y = self.np_random.integers(250, 380)
            self.platforms.append(pygame.Rect(p_x, p_y, p_w, 20))
            p_x += p_w + self.np_random.integers(self.min_platform_gap, self.max_platform_gap)
        
        self.player_pos[1] = self.platforms[1].top - self.PLAYER_HEIGHT
        self.player_pos[0] = self.platforms[1].centerx
        self.last_platform_landed = self.platforms[1]

        self.bonuses = []
        self._generate_initial_bonuses()

        self.finish_flag_rect = pygame.Rect(self.FINISH_LINE_X, self.platforms[-1].top - 60, 20, 60)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Player Input ---
        if movement == 3: self.player_vel[0] = -self.PLAYER_RUN_SPEED
        elif movement == 4: self.player_vel[0] = self.PLAYER_RUN_SPEED
        else: self.player_vel[0] = 0
            
        is_jump_action = (movement == 1 or space_held)
        if is_jump_action and self.player_on_ground:
            jump_power = self.PLAYER_LONG_JUMP_POWER if shift_held else self.PLAYER_JUMP_POWER
            self.player_vel[1] = jump_power
            self.player_on_ground = False
            for _ in range(10):
                self.particles.append({
                    'pos': [self.player_rect.centerx, self.player_rect.bottom],
                    'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(1, 3)],
                    'life': self.np_random.integers(10, 20),
                    'color': self.COLOR_PLAYER_GLOW[:3]
                })

        # --- Physics & World Update ---
        self.steps += 1
        self.timer -= 1
        
        if self.steps > 0 and self.steps % 500 == 0: self.scroll_speed += 0.01
        if self.steps > 0 and self.steps % 1000 == 0: self.min_platform_gap = min(200, self.min_platform_gap + 1)

        self.player_vel[1] += self.GRAVITY
        
        # FIX: The player's velocity from input is relative to the world.
        # To get the screen-space velocity, we must account for the world's scrolling.
        self.player_pos[0] += self.player_vel[0] - self.scroll_speed
        
        self.player_pos[1] += self.player_vel[1]
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.SCREEN_WIDTH - self.PLAYER_WIDTH - 20)
        self.player_rect.topleft = (int(self.player_pos[0]), int(self.player_pos[1]))
        self.player_anim_timer += 1

        self.world_scroll_x += self.scroll_speed
        
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] - self.scroll_speed
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

        # --- Collision Detection ---
        self.player_on_ground = False
        current_platform = None
        for plat in self.platforms:
            plat_screen_rect = plat.move(-self.world_scroll_x, 0)
            if self.player_rect.colliderect(plat_screen_rect) and self.player_vel[1] > 0 and self.player_rect.bottom - self.player_vel[1] <= plat_screen_rect.top + 1:
                self.player_pos[1] = plat_screen_rect.top - self.PLAYER_HEIGHT
                self.player_vel[1] = 0
                self.player_on_ground = True
                current_platform = plat
                break
        
        if self.player_on_ground:
            reward += 0.1
            if current_platform is not self.last_platform_landed:
                if self.last_platform_landed is not None:
                    gap = current_platform.left - self.last_platform_landed.right
                    reward -= 1.0 if gap > self.max_platform_gap * 0.8 else 0.2
                self.last_platform_landed = current_platform

        # --- Bonus Collection ---
        for bonus in self.bonuses[:]:
            bonus_screen_rect = bonus.move(-self.world_scroll_x, 0)
            if self.player_rect.colliderect(bonus_screen_rect):
                self.bonuses.remove(bonus)
                self.score += 10
                reward += 5
                for _ in range(20):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 5)
                    self.particles.append({
                        'pos': list(bonus_screen_rect.center),
                        'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                        'life': self.np_random.integers(15, 25),
                        'color': self.COLOR_BONUS
                    })

        # --- Platform Management ---
        self.platforms = [p for p in self.platforms if p.right - self.world_scroll_x > -50]
        last_plat = self.platforms[-1]
        if last_plat.right - self.world_scroll_x < self.SCREEN_WIDTH + 50 and last_plat.x < self.FINISH_LINE_X:
            p_x = last_plat.right + self.np_random.integers(self.min_platform_gap, self.max_platform_gap)
            p_w = self.np_random.integers(100, 300)
            max_y_diff = int(self.PLAYER_LONG_JUMP_POWER**2 / (2 * self.GRAVITY) * 0.6)
            p_y = self.np_random.integers(
                max(150, last_plat.y - max_y_diff),
                min(self.SCREEN_HEIGHT - 20, last_plat.y + max_y_diff)
            )
            new_plat = pygame.Rect(p_x, p_y, p_w, 20)
            self.platforms.append(new_plat)
            if self.np_random.random() < 0.3:
                self.bonuses.append(pygame.Rect(new_plat.centerx - 5, new_plat.top - 20, 10, 10))

        # --- Termination Conditions ---
        terminated = False
        if self.player_rect.top > self.SCREEN_HEIGHT:
            self.game_over, terminated, reward = True, True, -100
        if self.timer <= 0:
            self.game_over, terminated = True, True
        
        finish_screen_rect = self.finish_flag_rect.move(-self.world_scroll_x, 0)
        if self.player_rect.colliderect(finish_screen_rect):
            self.game_over, self.game_won, terminated = True, True, True
            reward = 100 + (self.timer / self.MAX_STEPS) * 20
            self.score += 1000
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.timer}

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        for p in self.particles:
            size = int(p['life'] / 3)
            if size > 0: pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        for plat in self.platforms:
            plat_screen_rect = plat.move(-self.world_scroll_x, 0)
            if plat_screen_rect.right > 0 and plat_screen_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_screen_rect, border_radius=3)
                edge_rect = pygame.Rect(plat_screen_rect.left, plat_screen_rect.top, plat_screen_rect.width, 5)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, edge_rect, border_top_left_radius=3, border_top_right_radius=3)

        for bonus in self.bonuses:
            bonus_screen_rect = bonus.move(-self.world_scroll_x, 0)
            if bonus_screen_rect.right > 0 and bonus_screen_rect.left < self.SCREEN_WIDTH:
                center_x, center_y = bonus_screen_rect.center; radius = bonus_screen_rect.width / 2
                points = []
                for i in range(5):
                    angle = math.radians(i * 72 - 90)
                    points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
                    angle = math.radians(i * 72 - 90 + 36)
                    points.append((center_x + radius * 0.4 * math.cos(angle), center_y + radius * 0.4 * math.sin(angle)))
                pygame.draw.polygon(self.screen, self.COLOR_BONUS, points)


        finish_screen_rect = self.finish_flag_rect.move(-self.world_scroll_x, 0)
        if finish_screen_rect.right > 0 and finish_screen_rect.left < self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, (200, 200, 200), (finish_screen_rect.left - 5, finish_screen_rect.top, 5, finish_screen_rect.height))
            flag_points = [finish_screen_rect.topleft, (finish_screen_rect.left + 30, finish_screen_rect.top + 15), finish_screen_rect.midleft]
            pygame.draw.polygon(self.screen, self.COLOR_FINISH, flag_points)

        if not (self.game_over and not self.game_won): self._render_player()

    def _render_player(self):
        p_rect = self.player_rect
        glow_center = (int(p_rect.centerx), int(p_rect.centery)); glow_radius = int(p_rect.width * 1.2)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius))

        body_rect = pygame.Rect(p_rect.left, p_rect.top + 10, p_rect.width, p_rect.height - 15)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
        head_center = (p_rect.centerx, p_rect.top + 8)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, head_center, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), (head_center[0] + 3, head_center[1]), 3)

        anim_offset = math.sin(self.player_anim_timer * 0.5) * 3 if self.player_on_ground and self.player_vel[0] != 0 else 0
        if not self.player_on_ground: anim_offset = -3
        leg_y = body_rect.bottom; leg_w = 6; leg_h = 5
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_rect.left + 3, leg_y, leg_w, leg_h + anim_offset), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_rect.right - 3 - leg_w, leg_y, leg_w, leg_h - anim_offset), border_radius=2)

    def _render_ui(self):
        self._draw_text(f"SCORE: {self.score}", (self.SCREEN_WIDTH - 10, 10), self.font_ui, self.COLOR_TEXT, align="topright")
        self._draw_text(f"TIME: {self.timer / 30:.1f}", (10, 10), self.font_ui, self.COLOR_TEXT, align="topleft")
        if self.game_over:
            msg, color = ("YOU WIN!", self.COLOR_FINISH) if self.game_won else ("GAME OVER", (255, 80, 80))
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_game_over, color, align="center")

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(**{align: pos})
        self.screen.blit(shadow_surf, (text_rect.left + 2, text_rect.top + 2))
        self.screen.blit(text_surf, text_rect)

    def _generate_initial_bonuses(self):
        for plat in self.platforms[2:]:
            if self.np_random.random() < 0.3:
                self.bonuses.append(pygame.Rect(plat.centerx - 5, plat.top - 20, 10, 10))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # For linux, use 'x11'. For windows, use 'windows'. For headless, use 'dummy'.
    # This is only for the human-playable demo.
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    # Re-initialize pygame with the new video driver for display
    pygame.quit()
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    done = False
    clock = pygame.time.Clock()
    
    print(env.game_description)
    print(env.user_guide)
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True
        
        clock.tick(30)
        
    print(f"Game Over. Final Score: {info['score']}")
    env.close()