import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ for air control. Press Space to jump. Hold Shift while jumping for a high jump."
    )

    game_description = (
        "Hop between procedurally generated platforms, collect coins, and reach the target score before falling or running out of time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 180
        self.WIN_SCORE = 20

        # Physics
        self.GRAVITY = 0.5
        self.FRICTION = -0.1
        self.AIR_FRICTION = 0.99
        self.PLAYER_ACCEL = 0.5
        self.PLAYER_MAX_SPEED = 5
        self.SHORT_JUMP_STRENGTH = -9
        self.HIGH_JUMP_STRENGTH = -12

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_PLATFORM = (150, 150, 170)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_COIN_GLOW = (255, 223, 0, 60)
        self.COLOR_BONUS_COIN = (255, 100, 255)
        self.COLOR_BONUS_COIN_GLOW = (255, 100, 255, 60)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_TIMER_BAR_FULL = (0, 255, 128)
        self.COLOR_TIMER_BAR_MID = (255, 255, 0)
        self.COLOR_TIMER_BAR_LOW = (255, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
            self.font_stage = pygame.font.SysFont('Consolas', 18)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 30)
            self.font_stage = pygame.font.Font(None, 24)

        # --- Internal State ---
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = self.MAX_TIME_SECONDS * self.FPS
        self.time_remaining_steps = 0
        self.stage = 1
        
        self.player_rect = None
        self.player_vel = None
        self.on_platform = False
        self.can_jump = False
        self.just_landed_platform_id = None

        self.platforms = deque()
        self.coins = deque()
        self.camera_offset = pygame.Vector2(0, 0)
        self.rightmost_generation_x = 0
        
        self.platform_speed_mod = 1.0
        self.platform_amp_mod = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Game state
        self.score = 0
        self.steps = 0
        self.time_remaining_steps = self.max_steps
        self.stage = 1
        self.game_over = False

        # Difficulty modifiers
        self.platform_speed_mod = 1.0
        self.platform_amp_mod = 1.0

        # Player state
        self.player_rect = pygame.Rect(100, 200, 20, 20)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_platform = False
        self.can_jump = False
        self.just_landed_platform_id = -1

        # World state
        self.platforms = deque()
        self.coins = deque()
        self.camera_offset = pygame.Vector2(0, 0)
        self.rightmost_generation_x = 0
        
        self._generate_initial_platforms()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        self.time_remaining_steps -= 1
        reward = 0

        self._update_stage()
        self._update_player(action)
        self._update_platforms()
        
        landed_reward = self._handle_platform_collisions()
        reward += landed_reward
        
        coin_reward, coins_collected = self._handle_coin_collisions()
        reward += coin_reward
        self.score += coins_collected

        self._update_camera()
        self._cull_and_generate_new_elements()

        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        truncated = self.steps >= self.max_steps
        if truncated:
            terminated = True


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Jumping ---
        if self.can_jump and space_pressed:
            # SFX: Jump
            jump_strength = self.HIGH_JUMP_STRENGTH if shift_held else self.SHORT_JUMP_STRENGTH
            self.player_vel.y = jump_strength
            self.on_platform = False
            self.can_jump = False
            self.just_landed_platform_id = -1

        # --- Apply Horizontal Movement (Air Control) ---
        if not self.on_platform:
            if movement == 3:  # Left
                self.player_vel.x -= self.PLAYER_ACCEL
            elif movement == 4: # Right
                self.player_vel.x += self.PLAYER_ACCEL
        
        # --- Apply Physics ---
        if self.on_platform:
            # Apply friction on ground
            self.player_vel.x += self.player_vel.x * self.FRICTION
            if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        else:
            # Apply gravity and air friction
            self.player_vel.y += self.GRAVITY
            self.player_vel.x *= self.AIR_FRICTION

        # Clamp horizontal speed
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Update position
        self.player_rect.x += self.player_vel.x
        self.player_rect.y += self.player_vel.y

    def _handle_platform_collisions(self):
        self.on_platform = False
        reward = 0
        
        # Sort platforms by y-position to handle landing correctly
        sorted_platforms = sorted(self.platforms, key=lambda p: p['rect'].y)

        for plat in sorted_platforms:
            plat_rect = plat['rect']
            if self.player_rect.colliderect(plat_rect) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if self.player_rect.bottom - self.player_vel.y <= plat_rect.top + 1:
                    self.player_rect.bottom = plat_rect.top
                    self.player_vel.y = 0
                    self.on_platform = True
                    self.can_jump = True
                    # Reward for landing on a new platform
                    if self.just_landed_platform_id != id(plat):
                        reward += 0.5 # Reward for exploring
                        self.just_landed_platform_id = id(plat)
                    break # Stop after first valid collision
        return reward

    def _handle_coin_collisions(self):
        reward = 0
        coins_collected = 0
        for coin in list(self.coins):
            if self.player_rect.colliderect(coin['rect']):
                # SFX: Coin collect
                reward += coin['value']
                coins_collected += 1
                self.coins.remove(coin)
        return reward, coins_collected

    def _update_platforms(self):
        time = self.steps * 0.05 * self.platform_speed_mod
        for plat in self.platforms:
            plat_rect = plat['rect']
            offset = math.sin(time + plat['phase']) * plat['amplitude'] * self.platform_amp_mod
            plat_rect.y = plat['base_y'] + offset

    def _update_camera(self):
        # Camera follows player, keeping them on the left third of the screen
        target_x = self.player_rect.centerx - self.SCREEN_WIDTH / 3
        # Smooth camera movement
        self.camera_offset.x += (target_x - self.camera_offset.x) * 0.1
        
        # Vertical camera follow, keeping player centered
        target_y = self.player_rect.centery - self.SCREEN_HEIGHT / 2
        self.camera_offset.y += (target_y - self.camera_offset.y) * 0.1


    def _cull_and_generate_new_elements(self):
        # Cull elements to the left of the camera
        cull_x = self.camera_offset.x - 100
        self.platforms = deque(p for p in self.platforms if p['rect'].right > cull_x)
        self.coins = deque(c for c in self.coins if c['rect'].right > cull_x)

        # Generate new elements to the right
        if self.rightmost_generation_x < self.camera_offset.x + self.SCREEN_WIDTH + 200:
            self._generate_new_platforms()

    def _generate_initial_platforms(self):
        # Create a safe starting area
        start_y = self.SCREEN_HEIGHT * 0.75
        for i in range(-2, 10):
            plat_rect = pygame.Rect(i * 100, start_y, self.rng.integers(80, 150), 20)
            self.platforms.append({
                'rect': plat_rect, 'base_y': start_y, 'amplitude': 0, 'phase': 0, 'id': i
            })
        self.player_rect.midbottom = self.platforms[3]['rect'].midtop
        self.rightmost_generation_x = self.platforms[-1]['rect'].right
        self._generate_new_platforms() # Populate the first screen

    def _generate_new_platforms(self):
        last_plat = self.platforms[-1]
        
        # Max jump distance estimation
        max_dx = self.PLAYER_MAX_SPEED * (abs(self.HIGH_JUMP_STRENGTH * 2 / self.GRAVITY)) * 0.9
        max_dy = abs(self.HIGH_JUMP_STRENGTH**2 / (2 * self.GRAVITY)) * 0.8

        while self.rightmost_generation_x < self.camera_offset.x + self.SCREEN_WIDTH + 400:
            dx = self.rng.uniform(60, max_dx)
            dy = self.rng.uniform(-max_dy, max_dy)
            
            new_x = last_plat['rect'].right + dx
            new_y = np.clip(last_plat['rect'].y + dy, 100, self.SCREEN_HEIGHT - 50)
            
            width = self.rng.integers(70, 140)
            plat_rect = pygame.Rect(new_x, new_y, width, 20)
            
            new_plat = {
                'rect': plat_rect,
                'base_y': new_y,
                'amplitude': self.rng.uniform(10, 50),
                'phase': self.rng.uniform(0, 2 * math.pi)
            }
            self.platforms.append(new_plat)
            
            # Place coins
            if self.rng.random() < 0.7: # 70% chance to have a coin
                is_bonus = self.rng.random() < 0.2 # 20% of coins are bonus
                coin_x = new_plat['rect'].centerx
                coin_y_offset = self.rng.uniform(-80, -40)
                # Riskier coins are higher up
                if is_bonus:
                    coin_y_offset -= 30

                self.coins.append({
                    'rect': pygame.Rect(coin_x - 10, new_plat['rect'].top + coin_y_offset, 20, 20),
                    'value': 2.0 if is_bonus else 1.0,
                    'is_bonus': is_bonus
                })
            
            last_plat = new_plat
            self.rightmost_generation_x = new_plat['rect'].right

    def _update_stage(self):
        time_elapsed_steps = self.steps
        if self.stage == 1 and time_elapsed_steps >= self.FPS * 60:
            self.stage = 2
            self.platform_speed_mod = 1.2
            self.platform_amp_mod = 1.1
        elif self.stage == 2 and time_elapsed_steps >= self.FPS * 120:
            self.stage = 3
            self.platform_speed_mod = 1.4
            self.platform_amp_mod = 1.2

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            # SFX: Win
            return True, 100.0  # Win condition
        if self.player_rect.top > self.SCREEN_HEIGHT + 100:
            # SFX: Fall
            return True, -10.0  # Fell off screen
        if self.time_remaining_steps <= 0:
            return True, -10.0  # Time out
        return False, 0.0

    def _get_observation(self):
        self._render_background()
        self._render_platforms()
        self._render_coins()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        grad_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT * 0.8)
        
        y = 0
        while y < grad_rect.height:
            interp = y / grad_rect.height
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            self.screen.fill(color, (0, y, self.SCREEN_WIDTH, 1))
            y += 1

    def _render_platforms(self):
        for plat in self.platforms:
            screen_rect = plat['rect'].move(-self.camera_offset.x, -self.camera_offset.y)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)

    def _render_coins(self):
        glow_radius = 15 + math.sin(self.steps * 0.2) * 3
        for coin in self.coins:
            screen_pos = coin['rect'].center - self.camera_offset
            if self.screen.get_rect().collidepoint(screen_pos):
                color = self.COLOR_BONUS_COIN if coin['is_bonus'] else self.COLOR_COIN
                glow_color = self.COLOR_BONUS_COIN_GLOW if coin['is_bonus'] else self.COLOR_COIN_GLOW
                
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(glow_radius), glow_color)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 10, color)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 10, color)

    def _render_player(self):
        screen_rect = self.player_rect.move(-self.camera_offset.x, -self.camera_offset.y)
        
        # Glow effect
        glow_center = screen_rect.center
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], 20, self.COLOR_PLAYER_GLOW)
        
        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_stage.render(f"STAGE: {self.stage}/3", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 40))

        # Timer bar
        timer_percent = max(0, self.time_remaining_steps / self.max_steps)
        bar_width = (self.SCREEN_WIDTH - 20) * timer_percent
        
        if timer_percent > 0.5:
            bar_color = self.COLOR_TIMER_BAR_FULL
        elif timer_percent > 0.2:
            bar_color = self.COLOR_TIMER_BAR_MID
        else:
            bar_color = self.COLOR_TIMER_BAR_LOW
        
        pygame.draw.rect(self.screen, (50,50,50), (10, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH - 20, 10))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, self.SCREEN_HEIGHT - 20, bar_width, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining_seconds": self.time_remaining_steps / self.FPS,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Note: Gymnasium's auto_advance is for training loops. For human play, we need a different loop.
    
    # Set this to False to play manually
    is_headless = False 

    if is_headless:
        # For testing the environment's stepping and rendering without a display
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        print("Initial state:", info)
        for _ in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.", info)
                obs, info = env.reset()
        env.close()
    else:
        # Human-playable version
        import os
        os.environ['SDL_VIDEODRIVER'] = "x11"
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 50)
        
        env = GameEnv(render_mode="rgb_array")
        env.auto_advance = False # Let the human control the frame rate
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        
        terminated = False
        truncated = False
        clock = pygame.time.Clock()
        
        while True:
            # --- Action mapping for human play ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            should_quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_quit = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False

            if should_quit:
                break

            if terminated or truncated:
                # Game over screen or logic
                pass
            else:
                 # --- Step the environment ---
                obs, reward, terminated, truncated, info = env.step(action)

            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(env.FPS)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                # Wait for a moment before allowing reset
                pygame.time.wait(1000)

        env.close()
        pygame.quit()