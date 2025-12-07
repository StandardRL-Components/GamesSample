import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to move on platforms. Use ↑↓←→ to aim your jump. "
        "Hold Shift for a short hop. Hold Space while jumping up for a power jump."
    )

    game_description = (
        "Hop between procedurally generated platforms to reach the top before time runs out "
        "in this fast-paced, arcade-style space hopper."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()

        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Constants ---
        self.GRAVITY = 0.4
        self.PLAYER_MOVE_SPEED = 4.0
        self.JUMP_VELOCITY = -9.0
        self.POWER_JUMP_MULTIPLIER = 1.4
        self.SHORT_JUMP_VELOCITY = -5.0
        self.AIR_CONTROL_FACTOR = 0.2
        self.AIR_FRICTION = 0.98
        self.LANDING_FRICTION = 0.5
        self.INITIAL_TIMER_SECONDS = 120

        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_EYE = (10, 10, 10)
        self.COLOR_PLATFORM = (100, 200, 255)
        self.COLOR_PLATFORM_BORDER = (60, 120, 153)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_TIMER_NORMAL = (0, 255, 128)
        self.COLOR_TIMER_WARNING = (255, 128, 0)
        self.COLOR_TIMER_CRITICAL = (255, 0, 0)

        # --- Fonts ---
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_platform = None
        self.can_jump = None
        self.player_facing_direction = None
        self.current_platform_index = None
        self.max_platform_index_reached = None
        self.player_squash = None
        self.platforms = None
        self.platform_oscillations = None
        self.stars = None
        self.particles = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.victory = None
        self.np_random = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.timer = self.INITIAL_TIMER_SECONDS * self.FPS

        self._generate_stars()
        self._generate_platforms()

        start_platform = self.platforms[0]['rect']
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - 15)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = pygame.Vector2(20, 30)
        self.on_platform = True
        self.can_jump = True
        self.player_facing_direction = 1  # 1:up, 2:down, 3:left, 4:right
        self.current_platform_index = 0
        self.max_platform_index_reached = 0

        self.particles = []
        self.player_squash = 1.0

        return self._get_observation(), self._get_info()

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            r = self.np_random.random() * 1.5
            self.stars.append((x, y, r))

    def _generate_platforms(self):
        self.platforms = []
        num_platforms = 12
        y_spacing = (self.SCREEN_HEIGHT - 100) / (num_platforms - 1)

        p_w, p_h = 150, 20
        p_x, p_y = (self.SCREEN_WIDTH - p_w) / 2, self.SCREEN_HEIGHT - 30
        rect = pygame.Rect(p_x, p_y, p_w, p_h)
        self.platforms.append({'rect': rect, 'base_y': p_y, 'osc_amp': 0, 'osc_freq': 0, 'osc_phase': 0})

        for i in range(1, num_platforms - 1):
            p_w = max(50, 120 - i * 6)
            p_y = self.SCREEN_HEIGHT - 30 - i * y_spacing
            prev_plat = self.platforms[-1]['rect']
            min_x = max(10, prev_plat.centerx - 150)
            max_x = min(self.SCREEN_WIDTH - p_w - 10, prev_plat.centerx + 150 - p_w)
            p_x = self.np_random.uniform(min_x, max_x)
            rect = pygame.Rect(p_x, p_y, p_w, p_h)
            self.platforms.append({
                'rect': rect, 'base_y': p_y, 'osc_amp': 0,
                'osc_freq': self.np_random.uniform(0.01, 0.03),
                'osc_phase': self.np_random.uniform(0, 2 * math.pi)
            })

        p_w, p_h = 80, 20
        p_y = 50
        prev_plat = self.platforms[-1]['rect']
        min_x = max(10, prev_plat.centerx - 150)
        max_x = min(self.SCREEN_WIDTH - p_w - 10, prev_plat.centerx + 150 - p_w)
        p_x = self.np_random.uniform(min_x, max_x)
        rect = pygame.Rect(p_x, p_y, p_w, p_h)
        self.platforms.append({'rect': rect, 'base_y': p_y, 'osc_amp': 0, 'osc_freq': 0, 'osc_phase': 0})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- 1. Update Game World ---
        self._update_world_state()

        # --- 2. Handle Input & Jumping ---
        if self.can_jump and (movement != 0 or shift_held or space_held):
            reward += self._handle_jump(movement, space_held, shift_held)

        # --- 3. Update Player Physics ---
        self._update_player_physics(movement)

        # --- 4. Handle Collisions ---
        if not self.on_platform and self.player_vel.y > 0:
            reward += self._handle_collisions()

        # --- 5. Finalize State & Check Termination ---
        terminated = False
        if self.player_pos.y > self.SCREEN_HEIGHT + self.player_size.y:
            terminated, self.game_over = True, True
            reward -= 20.0  # Penalty for falling
            # sfx: fall
        if self.timer <= 0 and not self.victory:
            terminated, self.game_over = True, True
            # sfx: time_up
        if self.victory:
            terminated = True
        
        if self.on_platform and not terminated:
            reward += 0.01 # Small survival reward

        self.score += reward
        self.score = max(0, self.score)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_world_state(self):
        self.steps += 1
        self.timer -= 1

        time_elapsed_sec = (self.INITIAL_TIMER_SECONDS * self.FPS - self.timer) / self.FPS
        oscillation_amplitude = 0.5 * (time_elapsed_sec // 30)
        for i in range(1, len(self.platforms) - 1):
            plat_info = self.platforms[i]
            plat_info['osc_amp'] = oscillation_amplitude
            offset = plat_info['osc_amp'] * math.sin(plat_info['osc_freq'] * self.steps + plat_info['osc_phase'])
            plat_info['rect'].y = plat_info['base_y'] + offset

        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_jump(self, movement, space_held, shift_held):
        reward = 0
        
        # If no directional input, but space/shift is held, default to up
        if movement == 0:
             self.player_facing_direction = 1
        else:
            self.player_facing_direction = movement
        
        jump_dir = pygame.Vector2(0, 0)
        jump_velocity = self.SHORT_JUMP_VELOCITY if shift_held else self.JUMP_VELOCITY

        if self.player_facing_direction == 1: # Up
            jump_dir.y = -1
            if space_held and not shift_held:
                jump_velocity *= self.POWER_JUMP_MULTIPLIER
                self._create_particles(20, self.player_pos, (255, 200, 100)) # sfx: power_jump
        elif self.player_facing_direction == 2: jump_dir.y = 0.2 # Down
        elif self.player_facing_direction == 3: jump_dir.x = -1 # Left
        elif self.player_facing_direction == 4: jump_dir.x = 1 # Right

        if jump_dir.length() > 0: jump_dir.normalize_ip()

        current_plat = self.platforms[self.current_platform_index]['rect']
        dist_from_edge = min(self.player_pos.x - current_plat.left, current_plat.right - self.player_pos.x)
        if dist_from_edge < current_plat.width * 0.15:
            reward -= 1.0  # Risky jump penalty

        self.player_vel.x = jump_dir.x * self.PLAYER_MOVE_SPEED * (0.6 if shift_held else 1.2)
        self.player_vel.y = jump_velocity if jump_dir.y < 0 else jump_dir.y * 4
        
        self.on_platform, self.can_jump = False, False
        self.player_squash = 1.5  # Stretch for jump
        self._create_particles(10, self.player_pos, (200, 200, 255)) # sfx: jump
        return reward

    def _update_player_physics(self, movement):
        if not self.on_platform:
            self.player_vel.y += self.GRAVITY
            move_x = 0
            if movement == 3: move_x = -1
            if movement == 4: move_x = 1
            self.player_vel.x += move_x * self.AIR_CONTROL_FACTOR
            self.player_vel.x *= self.AIR_FRICTION
        else:
            move_x = 0
            if movement == 3: move_x = -1
            if movement == 4: move_x = 1
            self.player_pos.x += move_x * self.PLAYER_MOVE_SPEED * 0.7
            current_plat = self.platforms[self.current_platform_index]['rect']
            self.player_pos.x = max(current_plat.left, min(self.player_pos.x, current_plat.right))
        
        self.player_pos += self.player_vel
        self.player_squash = max(1.0, self.player_squash - 0.05)

    def _handle_collisions(self):
        reward = 0
        player_rect = self._get_player_rect()
        # Use player's previous bottom y for more stable collision
        player_prev_bottom = self.player_pos.y - self.player_vel.y + player_rect.height / 2

        for i, plat_info in enumerate(self.platforms):
            plat = plat_info['rect']
            if player_rect.colliderect(plat) and player_prev_bottom <= plat.top:
                self.player_pos.y = plat.top - self.player_size.y / 2
                self.player_vel.y = 0
                self.player_vel.x *= self.LANDING_FRICTION
                self.on_platform, self.can_jump = True, True
                self.player_squash = 0.6 # Squash on land
                # sfx: land

                self.current_platform_index = i
                if i > self.max_platform_index_reached:
                    new_height_reward = 5.0 * (i - self.max_platform_index_reached)
                    reward += new_height_reward
                    self.max_platform_index_reached = i
                    self._create_particles(30, self.player_pos, (255, 255, 100)) # sfx: new_height

                if i == len(self.platforms) - 1:
                    self.victory, self.game_over = True, True
                    reward += 100.0 # sfx: victory
                break
        return reward

    def _get_player_rect(self):
        w = self.player_size.x / self.player_squash
        h = self.player_size.y * self.player_squash
        return pygame.Rect(self.player_pos.x - w/2, self.player_pos.y - h/2, w, h)

    def _create_particles(self, count, pos, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'color': color,
                'life': self.np_random.integers(10, 25),
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y, r in self.stars:
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(r), (200, 200, 220))

        for p in self.particles:
            alpha = int(255 * (p['life'] / 25))
            color_with_alpha = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color_with_alpha)
            except TypeError: # Handle potential color format issue
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])


        for i, plat_info in enumerate(self.platforms):
            plat_rect = plat_info['rect']
            color = self.COLOR_PLATFORM
            if i == len(self.platforms) - 1:
                pulse = (math.sin(self.steps * 0.1) + 1) / 2
                color = tuple(np.clip(np.array([150, 220, 150]) + np.array([105, -120, 105]) * pulse, 0, 255))
            
            pygame.draw.rect(self.screen, color, plat_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_BORDER, plat_rect, width=2, border_radius=3)

        player_rect = self._get_player_rect()
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(player_rect.width/3))
        
        eye_x_offset = 3 if self.player_facing_direction == 4 else -3
        if self.player_facing_direction in [1, 2]: eye_x_offset = 0
        eye_pos = (int(player_rect.centerx + eye_x_offset), int(player_rect.centery - player_rect.height*0.1))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, eye_pos, int(player_rect.width/6))

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        
        if time_left < 10: color = self.COLOR_TIMER_CRITICAL
        elif time_left < 30: color = self.COLOR_TIMER_WARNING
        else: color = self.COLOR_TIMER_NORMAL
            
        timer_surf = self.font_ui.render(timer_text, True, color)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))
        
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            s = pygame.Surface((msg_rect.width + 20, msg_rect.height + 20), pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (msg_rect.left - 10, msg_rect.top - 10))

            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
            "max_height_reached": self.max_platform_index_reached,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        
        # --- Pygame setup for human play ---
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Space Hopper")
        clock = pygame.time.Clock()
        
        total_reward = 0
        
        while not terminated:
            # --- Human Input ---
            movement = 0 # no-op
            space_held = 0
            shift_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Rendering ---
            # The observation is already a rendered frame, so we just need to display it.
            # Pygame uses (width, height), numpy uses (height, width), so we transpose.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

        print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
        env.close()
        pygame.quit()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("This is expected in a headless environment. The code is likely correct.")
        # Test headless functionality
        print("Testing headless functionality...")
        env = GameEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, float)
        env.close()
        print("Headless test passed.")