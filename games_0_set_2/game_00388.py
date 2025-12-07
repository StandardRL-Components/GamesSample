
# Generated: 2025-08-27T13:30:25.477052
# Source Brief: brief_00388.md
# Brief Index: 388

        
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
        "Controls: ←→ to run, ↑ to jump. Hold Shift to dash (costs 1 risk token)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced procedural platformer. Grab risk tokens to power your dash and reach the goal flag as fast as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 150, 150, 50)
    COLOR_PLATFORM = (100, 110, 120)
    COLOR_RISK_TOKEN = (255, 220, 0)
    COLOR_FLAG = (80, 220, 120)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (200, 200, 255)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Physics
    GRAVITY = 0.4
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.12
    PLAYER_MAX_SPEED = 6.0
    JUMP_STRENGTH = -10.0
    DASH_STRENGTH = 15.0

    # Game
    LEVEL_WIDTH_PIXELS = 5000
    MAX_STEPS = 1800 # 60 seconds at 30fps
    TIME_LIMIT_SECONDS = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Game state (persistent across resets)
        self.current_level = 0
        self.initial_max_platform_gap = 80
        self.max_platform_gap = self.initial_max_platform_gap
        
        # Will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        self.prev_shift_held = False
        self.platforms = []
        self.risk_tokens = []
        self.particles = []
        self.flag_rect = None
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.risk_tokens_collected = 0
        self.game_over = False
        self.game_outcome = ""
        self.last_player_x_for_reward = 0

        self.validate_implementation()

    def _generate_level(self):
        self.platforms.clear()
        self.risk_tokens.clear()
        
        # Start platform
        x = 20
        start_platform = pygame.Rect(x, self.SCREEN_HEIGHT - 50, 200, 50)
        self.platforms.append(start_platform)
        x += start_platform.width

        # Procedural generation loop
        while x < self.LEVEL_WIDTH_PIXELS - 300:
            gap = self.np_random.integers(40, self.max_platform_gap + 1)
            x += gap
            
            width = self.np_random.integers(80, 250)
            height = self.np_random.integers(
                max(50, self.platforms[-1].top - 100),
                min(self.SCREEN_HEIGHT - 20, self.platforms[-1].top + 100)
            )
            
            new_platform = pygame.Rect(x, height, width, self.SCREEN_HEIGHT - height)
            self.platforms.append(new_platform)

            # Add risk token?
            if self.np_random.random() < 0.3:
                token_pos = pygame.Vector2(new_platform.centerx, new_platform.top - 30)
                self.risk_tokens.append({'pos': token_pos, 'radius': 10, 'collected': False})

            x += width
        
        # Final platform and flag
        final_platform = self.platforms[-1]
        self.flag_rect = pygame.Rect(final_platform.right - 60, final_platform.top - 60, 40, 60)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(120, self.SCREEN_HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 24, 24)
        self.on_ground = False
        self.prev_shift_held = True # Prevent dash on first frame
        self.last_player_x_for_reward = self.player_pos.x

        self.steps = 0
        self.score = 0
        self.timer = self.TIME_LIMIT_SECONDS
        self.risk_tokens_collected = 0
        self.particles.clear()
        self.camera_x = 0.0
        
        self.game_over = False
        self.game_outcome = ""

        # Difficulty scaling
        win_condition_met = options and options.get("win", False)
        if win_condition_met:
            self.current_level += 1
            self.max_platform_gap = min(130, self.initial_max_platform_gap + 5 * self.current_level)
        else: # On loss or first game
            self.current_level = 0
            self.max_platform_gap = self.initial_max_platform_gap
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # When the game is over, we still need to return a valid tuple.
            # The state doesn't change, reward is 0.
            # The 'terminated' flag is already True from the previous step that ended the game.
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.timer -= 1.0 / 30.0 # Assuming 30 FPS

        # 1. Unpack Action
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        shift_held = action[2] == 1

        # 2. Handle Input and Player Logic
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
        
        # Friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump
            for _ in range(5):
                self.particles.append(self._create_particle(self.player_rect.midbottom, life=15, angle_range=(240, 300), speed_range=(1, 3)))

        # Dashing
        dash_triggered = shift_held and not self.prev_shift_held
        if dash_triggered and self.risk_tokens_collected > 0:
            self.risk_tokens_collected -= 1
            direction = 1 if self.player_vel.x >= 0 else -1
            self.player_vel.x = self.DASH_STRENGTH * direction
            reward += 5 # Dash reward
            # Sound: Dash
            for i in range(20):
                p_pos = self.player_rect.center - pygame.Vector2(direction * i * 1.5, 0)
                self.particles.append(self._create_particle(p_pos, life=20, angle_range=(0, 360), speed_range=(0.5, 2)))

        self.prev_shift_held = shift_held

        # 3. Physics Update
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(15, self.player_vel.y) # Terminal velocity

        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        
        # Horizontal collisions
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0: self.player_rect.right = plat.left
                elif self.player_vel.x < 0: self.player_rect.left = plat.right
                self.player_pos.x = self.player_rect.x
                self.player_vel.x = 0

        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)
        self.on_ground = False

        # Vertical collisions
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0:
                    self.player_rect.bottom = plat.top
                    self.on_ground = True
                elif self.player_vel.y < 0:
                    self.player_rect.top = plat.bottom
                self.player_pos.y = self.player_rect.y
                self.player_vel.y = 0

        # 4. Collect Risk Tokens
        for token in self.risk_tokens[:]:
            if not token['collected']:
                token_rect = pygame.Rect(token['pos'].x - token['radius'], token['pos'].y - token['radius'], token['radius']*2, token['radius']*2)
                if self.player_rect.colliderect(token_rect):
                    token['collected'] = True
                    self.risk_tokens_collected += 1
                    reward += 1.0
                    # Sound: Token Collect
                    for _ in range(10):
                        self.particles.append(self._create_particle(token['pos'], life=15, angle_range=(0, 360), speed_range=(2, 4), color=self.COLOR_RISK_TOKEN))

        # 5. Update Camera
        self.camera_x += (self.player_rect.centerx - self.camera_x - self.SCREEN_WIDTH / 2) * 0.1

        # 6. Calculate Progress Reward
        dist_to_flag = self.flag_rect.centerx - self.player_pos.x
        last_dist_to_flag = self.flag_rect.centerx - self.last_player_x_for_reward
        if dist_to_flag < last_dist_to_flag: reward += 0.1
        else: reward -= 0.1
        self.last_player_x_for_reward = self.player_pos.x
        
        # 7. Check Termination
        terminated = False
        win = False
        if self.player_rect.colliderect(self.flag_rect):
            terminated, win, reward_mod, self.game_outcome = True, True, 60, "LEVEL COMPLETE!"
        elif self.player_rect.top > self.SCREEN_HEIGHT:
            terminated, reward_mod, self.game_outcome = True, -10, "YOU FELL!"
        elif self.timer <= 0:
            terminated, reward_mod, self.game_outcome = True, -5, "TIME UP!"
        elif self.steps >= self.MAX_STEPS:
            terminated, reward_mod, self.game_outcome = True, 0, "MAX STEPS REACHED"

        if terminated:
            reward += reward_mod
            self.game_over = True
            # The next call to reset() will use this 'win' status
            self.reset_options = {"win": win}
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle(self, pos, life, angle_range, speed_range, color=COLOR_PARTICLE):
        angle = math.radians(self.np_random.uniform(angle_range[0], angle_range[1]))
        speed = self.np_random.uniform(speed_range[0], speed_range[1])
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        return {'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color}

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                size = int(5 * (p['life'] / p['max_life']))
                if size > 0:
                    draw_pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, (*p['color'], alpha), (size, size), size)
                    self.screen.blit(temp_surf, (draw_pos[0] - size, draw_pos[1] - size))

    def _get_observation(self):
        # 1. Draw Background Gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # 2. Draw Game Elements
        for plat in self.platforms:
            draw_rect = plat.move(-self.camera_x, 0)
            if draw_rect.right > 0 and draw_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect)
        
        for token in self.risk_tokens:
            if not token['collected']:
                draw_pos = (int(token['pos'].x - self.camera_x), int(token['pos'].y))
                if -20 < draw_pos[0] < self.SCREEN_WIDTH + 20:
                    pygame.gfxdraw.aacircle(self.screen, draw_pos[0], draw_pos[1], token['radius'], self.COLOR_RISK_TOKEN)
                    pygame.gfxdraw.filled_circle(self.screen, draw_pos[0], draw_pos[1], token['radius'], self.COLOR_RISK_TOKEN)
        
        draw_flag_rect = self.flag_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, draw_flag_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(draw_flag_rect.left, draw_flag_rect.top), (draw_flag_rect.left - 20, draw_flag_rect.top + 10), (draw_flag_rect.left, draw_flag_rect.top + 20)])

        self._update_and_draw_particles()

        draw_player_rect = self.player_rect.move(-self.camera_x, 0)
        glow_surf = pygame.Surface((self.player_rect.width * 2, self.player_rect.height * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (draw_player_rect.x - self.player_rect.width/2, draw_player_rect.y - self.player_rect.height/2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, draw_player_rect, border_radius=4)

        # 3. Draw UI
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        timer_text = self.font_ui.render(f"TIME: {max(0, int(self.timer))}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        tokens_text = self.font_ui.render(f"DASH: {self.risk_tokens_collected}", True, self.COLOR_RISK_TOKEN)
        self.screen.blit(tokens_text, (10, 35))

        if self.game_over and self.game_outcome:
            outcome_text = self.font_game_over.render(self.game_outcome, True, self.COLOR_UI_TEXT)
            text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(outcome_text, text_rect)

        # 4. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "level": self.current_level,
            "risk_tokens": self.risk_tokens_collected
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption(env.game_description)
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH * 2, env.SCREEN_HEIGHT * 2))
    
    obs, info = env.reset()
    terminated = False
    
    key_map = {pygame.K_LEFT: 3, pygame.K_RIGHT: 4, pygame.K_UP: 1}
    
    running = True
    while running:
        action = [0, 0, 0]
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]: action[0] = key_map[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]: action[0] = key_map[pygame.K_RIGHT]
        
        if keys[pygame.K_UP]: action[0] = key_map[pygame.K_UP]
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf, real_screen.get_size())
        real_screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}. Resetting...")
            pygame.time.wait(2000)
            # Use the options from the previous game state to inform the reset
            obs, info = env.reset(seed=None, options=getattr(env, 'reset_options', None))
            terminated = False

        env.clock.tick(30)
        
    env.close()