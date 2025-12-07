import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move. Press ↑ or Space to jump. Reach the green finish line before time runs out!"
    )

    game_description = (
        "Ascend is a fast-paced platformer where you race against the clock to climb a tower of procedurally "
        "generated platforms. The world scrolls upwards, demanding quick reflexes and precise jumps."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG_TOP = (4, 12, 48)
    COLOR_BG_BOTTOM = (25, 66, 128)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_PLATFORM = (150, 160, 180)
    COLOR_PLATFORM_SHADOW = (100, 110, 130)
    COLOR_FINISH_LINE = (0, 255, 128)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 150)

    # Physics
    PLAYER_SIZE = 18
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.85
    MAX_PLAYER_SPEED = 7
    GRAVITY = 0.7
    JUMP_STRENGTH = -14

    # Gameplay
    PLATFORM_HEIGHT = 12
    LEVEL_HEIGHT_PLATFORMS = 40
    INITIAL_SCROLL_SPEED = 1.0
    SCROLL_ACCELERATION = 0.05 / 500 # Per brief

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.render_mode = render_mode
        self.background_surf = self._create_gradient_background()

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.time_left = 0
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.is_grounded = False
        self.camera_y = 0
        self.scroll_speed = 0
        self.platforms = []
        self.finish_line_y = 0
        self.particles = []
        self.start_y = 0
        self.highest_y_reached = 0
        self.highest_platform_y = 0
        
        # self.reset() is called by the test harness, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS

        # This places the player on the starting platform.
        # The platform's top is at SCREEN_HEIGHT - 40.
        # The player's position is its top coordinate.
        start_platform_y = self.SCREEN_HEIGHT - 40
        self.player_pos = [self.SCREEN_WIDTH / 2, start_platform_y - self.PLAYER_SIZE]
        self.player_vel = [0, 0]
        self.is_grounded = True
        self.scroll_speed = self.INITIAL_SCROLL_SPEED

        self.start_y = self.player_pos[1]
        self.highest_y_reached = self.player_pos[1]
        self.highest_platform_y = self.player_pos[1]
        
        self._generate_platforms()

        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.7
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms = []
        # Start platform
        start_plat = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 40, 200, self.PLATFORM_HEIGHT
        )
        self.platforms.append(start_plat)
        
        last_plat = start_plat
        current_y = start_plat.y

        for _ in range(self.LEVEL_HEIGHT_PLATFORMS):
            y_spacing = self.np_random.integers(60, int(self.PLAYER_SIZE + abs(self.JUMP_STRENGTH)**2 / (2 * self.GRAVITY)) - 20)
            current_y -= y_spacing
            
            width = self.np_random.integers(80, 180)
            
            max_horiz_dist = 220
            x_offset = self.np_random.integers(-max_horiz_dist, max_horiz_dist)
            
            x = last_plat.centerx - width / 2 + x_offset
            x = np.clip(x, 20, self.SCREEN_WIDTH - width - 20)

            new_plat = pygame.Rect(x, current_y, width, self.PLATFORM_HEIGHT)
            self.platforms.append(new_plat)
            last_plat = new_plat
        
        self.finish_line_y = last_plat.y - 80

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        
        # --- Handle Input & Player Physics ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_intent = 0
        if movement == 3: move_intent = -1  # Left
        elif movement == 4: move_intent = 1 # Right
        
        self.player_vel[0] += move_intent * self.PLAYER_ACCEL
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_PLAYER_SPEED, self.MAX_PLAYER_SPEED)
        
        wants_to_jump = (movement == 1) or space_held
        if wants_to_jump and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            self._create_jump_particles(self.player_pos)

        self.player_vel[1] += self.GRAVITY
        
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH - self.PLAYER_SIZE)

        # --- World Update ---
        self.scroll_speed += self.SCROLL_ACCELERATION
        self.player_pos[1] += self.scroll_speed
        for plat in self.platforms:
            plat.y += self.scroll_speed
        self.finish_line_y += self.scroll_speed

        # --- Collisions & Rewards ---
        landing_reward = self._handle_collisions()

        # --- Update Camera & Particles ---
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.6
        self.camera_y += (target_cam_y - self.camera_y) * 0.1
        self._update_particles()
        
        # --- Termination & Final Rewards ---
        terminated = False
        terminal_reward = 0
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + 50:
            terminated = True
            terminal_reward = -100 # Fell off screen
        elif self.time_left <= 0:
            terminated = True
            terminal_reward = -100 # Time out
        elif self.player_pos[1] + self.PLAYER_SIZE < self.finish_line_y:
            terminated = True
            terminal_reward = 100 # Reached goal
        
        self.game_over = terminated

        # --- Continuous Rewards ---
        continuous_reward = 0
        self.highest_y_reached = min(self.highest_y_reached, self.player_pos[1])
        if self.player_pos[1] < self.start_y:
            continuous_reward += 0.1
        if self.player_pos[1] > self.highest_platform_y and not self.is_grounded:
             continuous_reward -= 0.01

        reward = landing_reward + terminal_reward + continuous_reward
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        # Use player velocity from *before* it was changed by gravity in this step for a more stable check
        prev_player_bottom = player_rect.bottom - self.player_vel[1]
        
        landing_reward = 0
        on_any_platform = False

        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel[1] > 0 and prev_player_bottom <= plat.top + 1: # Add tolerance
                self.player_pos[1] = plat.top - self.PLAYER_SIZE
                self.player_vel[1] = 0
                on_any_platform = True
                
                if not self.is_grounded: # First frame of landing
                    self._create_land_particles((player_rect.centerx, plat.top))
                    if plat.top < self.highest_platform_y:
                        landing_reward += 1.0
                        self.highest_platform_y = plat.top
                break
        
        self.is_grounded = on_any_platform
        return landing_reward

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw finish line
        finish_y_on_screen = int(self.finish_line_y - self.camera_y)
        if 0 < finish_y_on_screen < self.SCREEN_HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (0, finish_y_on_screen), (self.SCREEN_WIDTH, finish_y_on_screen), 3)

        # Draw platforms
        for plat in self.platforms:
            plat_on_screen = plat.move(0, -self.camera_y)
            if plat_on_screen.top < self.SCREEN_HEIGHT and plat_on_screen.bottom > 0:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, plat_on_screen.move(0, 4))
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_on_screen)

        # Draw particles
        for p in self.particles:
            p_pos_on_screen = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            pygame.draw.circle(self.screen, p['color'], p_pos_on_screen, int(p['size']))

        # Draw player
        player_screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y))
        
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*self.COLOR_PLAYER_GLOW, 50)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_screen_pos[0] + self.PLAYER_SIZE/2 - glow_radius, player_screen_pos[1] + self.PLAYER_SIZE/2 - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        player_rect = pygame.Rect(player_screen_pos[0], player_screen_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_left // self.FPS):02d}"
        self._draw_text(time_text, self.font_small, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, self.font_small, (self.SCREEN_WIDTH / 2, 10), center=True)

        # Stage
        stage_text = f"STAGE: {self.stage}"
        self._draw_text(stage_text, self.font_small, (self.SCREEN_WIDTH - 10, 10), right=True)

    def _draw_text(self, text, font, pos, center=False, right=False):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.midtop = pos
        elif right:
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, text_rect.move(2, 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "stage": self.stage,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def _create_gradient_background(self):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.gfxdraw.hline(surf, 0, self.SCREEN_WIDTH, y, color)
        return surf

    def _create_jump_particles(self, pos):
        for _ in range(10):
            self.particles.append({
                'pos': [pos[0] + self.PLAYER_SIZE / 2, pos[1] + self.PLAYER_SIZE],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(0.5, 2.5)],
                'size': self.np_random.uniform(2, 5),
                'life': 20,
                'color': self.COLOR_PLATFORM,
            })

    def _create_land_particles(self, pos):
        for _ in range(15):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-2.5, 2.5), self.np_random.uniform(-2, 0)],
                'size': self.np_random.uniform(2, 6),
                'life': 25,
                'color': self.COLOR_PLATFORM,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['size'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
        
    def validate_implementation(self):
        # This is a helper for development and not used by the tests.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test reset
            obs, info = self.reset()
            # Test observation space  
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert obs.dtype == np.uint8
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert trunc == False
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except Exception as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment.
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ascend")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # --- Main Game Loop for Human Play ---
    while not done:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            done = True

        clock.tick(env.FPS)

    env.close()