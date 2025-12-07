
# Generated: 2025-08-28T06:02:41.239819
# Source Brief: brief_05764.md
# Brief Index: 5764

        
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


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, vx, vy, color, radius, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # A bit of gravity
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface):
        if self.lifetime <= 0:
            return
        
        alpha = int(255 * (self.lifetime / self.initial_lifetime))
        
        # Use a temporary surface for alpha blending to avoid affecting other draws
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            temp_surf,
            (*self.color, alpha),
            (self.radius, self.radius),
            self.radius
        )
        surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys for short jumps. Space for a long jump upwards. "
        "Shift + Left/Right for diagonal jumps."
    )

    game_description = (
        "Leap across procedurally generated platforms to reach the top. "
        "You have 3 lives and 30 seconds per stage. Clear 3 stages to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_FALLS = 3
        self.TIME_PER_STAGE = 30
        self.MAX_STEPS = 3 * self.TIME_PER_STAGE * self.FPS + 100 # Max time for 3 stages + buffer

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (5, 5, 15)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_FALL_X = (220, 50, 50)
        self.PLATFORM_COLORS = [
            (50, 205, 50), (30, 144, 255), (255, 69, 0), 
            (218, 112, 214), (0, 255, 255), (255, 215, 0)
        ]
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.falls = 0
        self.stage = 1
        self.game_over = False
        self.win = False
        
        self.time_left_in_stage = self.TIME_PER_STAGE * self.FPS
        self.max_height_reached = self.HEIGHT
        
        self.is_jumping = False
        self.jump_start_pos = (0, 0)
        self.jump_end_pos = (0, 0)
        self.jump_duration = 15
        self.jump_timer = 0
        self.just_landed = False
        
        self.particles = []
        
        self._generate_stage()
        
        return self._get_observation(), self._get_info()

    def _generate_stage(self):
        self.platforms = []
        plat_w, plat_h = 80, 15
        
        # 1. Create start platform
        start_plat = pygame.Rect(self.WIDTH // 2 - plat_w // 2, self.HEIGHT - 40, plat_w, plat_h)
        
        # 2. Generate a guaranteed path
        difficulty_mod = 1.0 + (self.stage - 1) * 0.3
        max_jump_h = 100 * difficulty_mod
        max_jump_v = 130 * difficulty_mod
        
        current_plat = start_plat
        path_platforms = [start_plat]
        
        # Generate a path upwards until it's in the top quarter of the screen
        while current_plat.centery > self.HEIGHT / 4:
            px, py = current_plat.center
            
            dx = self.np_random.uniform(-max_jump_h, max_jump_h)
            dy = self.np_random.uniform(-max_jump_v, -max_jump_v * 0.5)
            
            nx = np.clip(px + dx, plat_w // 2, self.WIDTH - plat_w // 2)
            ny = np.clip(py + dy, plat_h, self.HEIGHT - plat_h)
            
            new_plat = pygame.Rect(nx - plat_w // 2, ny - plat_h // 2, plat_w, plat_h)
            
            if not any(new_plat.colliderect(p) for p in path_platforms):
                path_platforms.append(new_plat)
                current_plat = new_plat
        
        self.platforms.extend(path_platforms)
        
        # 3. Add distractor platforms
        num_distractors = self.np_random.integers(5, 10)
        for _ in range(num_distractors):
            rand_x = self.np_random.integers(0, self.WIDTH - plat_w)
            rand_y = self.np_random.integers(40, self.HEIGHT - 60)
            distractor = pygame.Rect(rand_x, rand_y, plat_w, plat_h)
            if not any(distractor.colliderect(p) for p in self.platforms):
                self.platforms.append(distractor)
        
        self._reset_player_position()

    def _reset_player_position(self):
        start_plat = self.platforms[0]
        self.player_pos = start_plat.center
        self.player_size = 20
        self.is_jumping = False
        self.max_height_reached = self.player_pos[1]

    def step(self, action):
        reward = -0.01 # Small time penalty per frame
        self.steps += 1
        self.time_left_in_stage -= 1
        
        self._update_particles()
        self._update_jump_animation()

        # Landing consequences are processed after the jump animation is complete
        if self.just_landed:
            self.just_landed = False
            landing_reward = self._handle_landing()
            reward += landing_reward

        # Player can only act if not in the middle of a jump
        if not self.is_jumping:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            jump_type = None
            if space_held:
                jump_type = ('long_up', None)
            elif shift_held and movement in [3, 4]: # Left, Right
                jump_type = ('diagonal', movement)
            elif movement != 0:
                jump_type = ('short', movement)
            
            if jump_type:
                target_plat = self._find_jump_target(*jump_type)
                if target_plat:
                    # Sound: Jump
                    self._start_jump(target_plat.center)
                    reward += 1.0 # Reward for initiating a valid jump
                elif movement or space_held or shift_held: # An action was attempted but failed
                    # Sound: Fall_Start
                    fall_x = self.player_pos[0] + self.np_random.uniform(-30, 30)
                    fall_y = self.HEIGHT + self.player_size
                    self._start_jump((fall_x, fall_y))
        
        # Check termination conditions
        terminated = False
        if self.falls >= self.MAX_FALLS or self.time_left_in_stage <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        if self.game_over and not self.win:
            terminated = True
            reward -= 100.0 # Big loss penalty
        elif self.win:
            terminated = True
            reward += 100.0 # Big win reward

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_landing(self):
        reward = 0
        # Fall check
        if self.player_pos[1] >= self.HEIGHT + self.player_size / 2:
            self.falls += 1
            reward -= 5.0
            # Sound: Fall_Impact
            self._create_particles(self.jump_start_pos, self.COLOR_FALL_X, 40, 2)
            if self.falls < self.MAX_FALLS:
                self._reset_player_position()
        else: # Successful landing
            # Sound: Land
            self._create_particles(self.player_pos, self.COLOR_UI_TEXT, 15, 1)
            
            # New height reward
            if self.player_pos[1] < self.max_height_reached:
                reward += 5.0
                self.max_height_reached = self.player_pos[1]
                
            # Stage win check (on highest platform)
            goal_plat = min(self.platforms, key=lambda p: p.top)
            player_rect = pygame.Rect(self.player_pos[0] - self.player_size/2, self.player_pos[1] - self.player_size/2, self.player_size, self.player_size)
            
            if player_rect.colliderect(goal_plat):
                reward += 20.0
                self.stage += 1
                if self.stage > 3:
                    self.win = True
                    self.game_over = True
                else:
                    # Sound: Stage_Clear
                    self.time_left_in_stage = self.TIME_PER_STAGE * self.FPS
                    self._generate_stage()
        return reward

    def _start_jump(self, target_pos):
        self.is_jumping = True
        self.jump_timer = 0
        self.jump_start_pos = self.player_pos
        self.jump_end_pos = target_pos
        self._create_particles(self.player_pos, self.COLOR_PLAYER, 20, 1.5)

    def _update_jump_animation(self):
        if not self.is_jumping:
            return

        self.jump_timer += 1
        progress = min(1.0, self.jump_timer / self.jump_duration)
        eased_progress = 1 - (1 - progress) ** 3 # Ease-out cubic

        if progress >= 1.0:
            self.player_pos = self.jump_end_pos
            self.is_jumping = False
            self.just_landed = True
        else:
            start_x, start_y = self.jump_start_pos
            end_x, end_y = self.jump_end_pos
            
            current_x = start_x + (end_x - start_x) * eased_progress
            
            arc_height = max(0, start_y - end_y) * 0.4 + 30
            arc = math.sin(progress * math.pi) * arc_height
            current_y = start_y + (end_y - start_y) * eased_progress
            
            self.player_pos = (current_x, current_y - arc)

    def _find_jump_target(self, jump_type, movement=None):
        short_dist, long_dist, diag_dist = 120, 220, 180
        directions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        px, py = self.player_pos
        
        best_target, min_score = None, float('inf')

        for plat in self.platforms:
            if plat.collidepoint(px, py): continue

            cx, cy = plat.center
            dist = math.hypot(cx - px, cy - py)
            if dist == 0: continue

            vec_x, vec_y = (cx - px) / dist, (cy - py) / dist
            
            is_valid = False
            score = float('inf')

            if jump_type == 'long_up' and dist < long_dist and vec_y < -0.7:
                is_valid = True
                score = dist # Prefer closer long jumps
            elif jump_type == 'short' and movement:
                dir_vec = directions[movement]
                dot_product = vec_x * dir_vec[0] + vec_y * dir_vec[1]
                if dist < short_dist and dot_product > 0.8: # Strong alignment
                    is_valid = True
                    score = (1 - dot_product) * dist # Penalize misalignment
            elif jump_type == 'diagonal' and movement in [3, 4]: # Left/Right
                target_vx = -1 if movement == 3 else 1
                if dist < diag_dist and np.sign(vec_x) == target_vx and vec_y < -0.2:
                    is_valid = True
                    score = dist
            
            if is_valid and score < min_score:
                min_score = score
                best_target = plat
                
        return best_target

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = tuple(
                int(top * (1 - ratio) + bot * ratio)
                for top, bot in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        for i, plat in enumerate(self.platforms):
            color = self.PLATFORM_COLORS[i % len(self.PLATFORM_COLORS)]
            pygame.draw.rect(self.screen, color, plat, border_radius=3)
            highlight_color = tuple(min(255, c + 30) for c in color)
            pygame.draw.rect(self.screen, highlight_color, plat.inflate(-6, -6), border_radius=3)

        for p in self.particles:
            p.draw(self.screen)
            
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        ps = self.player_size
        player_rect = pygame.Rect(px - ps // 2, py - ps // 2, ps, ps)

        for i in range(ps // 2, 0, -2):
            alpha = 80 * (1 - (i / (ps // 2)))
            pygame.gfxdraw.aacircle(self.screen, px, py, i, (*self.COLOR_PLAYER_GLOW, int(alpha)))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        stage_text = self.font_main.render(f"Stage: {self.stage}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        time_sec = max(0, self.time_left_in_stage // self.FPS)
        timer_text = self.font_main.render(f"Time: {time_sec}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        for i in range(self.MAX_FALLS):
            pos = (self.WIDTH - 30 * (i + 1), self.HEIGHT - 30)
            if i < self.falls:
                pygame.draw.line(self.screen, self.COLOR_FALL_X, (pos[0]-10, pos[1]-10), (pos[0]+10, pos[1]+10), 4)
                pygame.draw.line(self.screen, self.COLOR_FALL_X, (pos[0]+10, pos[1]-10), (pos[0]-10, pos[1]+10), 4)
            else:
                pygame.draw.circle(self.screen, self.COLOR_UI_TEXT, pos, 10, 2)
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_main.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "falls": self.falls,
            "time_left_in_stage": self.time_left_in_stage // self.FPS
        }

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            radius = self.np_random.integers(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos[0], pos[1], vx, vy, color, radius, lifetime))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Leap Arcade")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()