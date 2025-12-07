import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Press Space to jump. Use left/right actions to move."
    game_description = "A minimalist neon platformer. Time your jumps across procedurally generated gaps and reach the goal."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

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

        # Colors
        self.COLOR_BG_TOP = (20, 10, 40)
        self.COLOR_BG_BOTTOM = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_PLATFORM = (60, 60, 220)
        self.COLOR_PLATFORM_BORDER = (120, 120, 255)
        self.COLOR_FLAG = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Physics constants
        self.GRAVITY = 0.7
        self.PLAYER_JUMP_STRENGTH = -14
        self.PLAYER_X_SPEED = 4.0
        self.PLAYER_SIZE = 20
        self.CAMERA_LERP_RATE = 0.1

        # Pre-render background
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

        # Game state variables
        self.level = 0
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.platforms = None
        self.flag_pos = None
        self.camera_x = None
        self.particles = None
        self.prev_space_held = None
        self.current_platform_index = None
        self.timer = None
        self.max_steps = 1500
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.prev_space_held = False
        self.particles = []

        if not hasattr(self, 'persistent_level'):
            self.persistent_level = 1
        self.level = self.persistent_level

        self._generate_platforms()

        start_platform = self.platforms[0]
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_SIZE]
        self.player_vel = [0, 0]  # FIX: Player starts with zero velocity
        self.is_grounded = True
        self.current_platform_index = 0
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 4

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        move_action = action[0]
        space_held = action[1] == 1

        # FIX: Control horizontal movement with action[0]
        if move_action == 1:  # Right
            self.player_vel[0] = self.PLAYER_X_SPEED
        elif move_action == 2:  # Left
            self.player_vel[0] = -self.PLAYER_X_SPEED
        else:  # 0, 3, 4 are no horizontal movement
            self.player_vel[0] = 0

        # Jumping
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and self.is_grounded:
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.is_grounded = False

        self.prev_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        self.timer += 1 / 30.0
        
        reward += 0.01  # Base reward for staying alive

        # Update player physics
        self.player_vel[1] += self.GRAVITY
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Collision detection
        self.is_grounded = False
        if self.player_vel[1] >= 0:  # Check for landing if falling or on ground
            for i, platform in enumerate(self.platforms):
                if player_rect.colliderect(platform) and player_rect.bottom <= platform.centery + 1:
                    self.player_pos[1] = platform.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.is_grounded = True
                    if self.current_platform_index != i:
                        self.current_platform_index = i
                        reward += 1.0
                        self._create_landing_particles(player_rect.midbottom)
                    break
        
        # Update camera
        target_camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 4
        self.camera_x += (target_camera_x - self.camera_x) * self.CAMERA_LERP_RATE

        # Update particles
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            terminated = True
            reward -= 10.0
        
        flag_rect = pygame.Rect(self.flag_pos[0], self.flag_pos[1] - 40, 30, 40)
        if player_rect.colliderect(flag_rect):
            terminated = True
            reward += 100.0
            self.persistent_level += 1

        truncated = self.steps >= self.max_steps
        
        if terminated or truncated:
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_platforms(self):
        self.platforms = []
        current_x = 50
        current_y = 300
        
        start_width = 150
        self.platforms.append(pygame.Rect(current_x, current_y, start_width, 200))
        current_x += start_width

        num_platforms = 15
        difficulty_mod = 1.0 + 0.05 * (self.level - 1)

        for i in range(num_platforms):
            gap_x = self.np_random.uniform(60, 110) * difficulty_mod
            gap_y = self.np_random.uniform(-70, 70)
            width = self.np_random.uniform(80, 200)
            
            current_x += gap_x
            next_y = current_y + gap_y
            next_y = np.clip(next_y, 150, self.SCREEN_HEIGHT - 50)
            
            self.platforms.append(pygame.Rect(current_x, next_y, width, 200))
            current_x += width
            current_y = next_y

        last_platform = self.platforms[-1]
        self.flag_pos = (last_platform.centerx, last_platform.top)

    def _create_landing_particles(self, pos):
        for _ in range(15):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-4, -1)]
            life = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        cam_x = int(self.camera_x)
        
        for p in self.platforms:
            p_on_screen = p.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p_on_screen, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_BORDER, p_on_screen, 2, border_radius=4)
            
        flag_x, flag_y = self.flag_pos
        flag_pole = pygame.Rect(flag_x - cam_x, flag_y - 40, 4, 40)
        flag_triangle = [(flag_x - cam_x, flag_y - 40), (flag_x - cam_x + 25, flag_y - 30), (flag_x - cam_x, flag_y - 20)]
        pygame.draw.rect(self.screen, self.COLOR_TEXT, flag_pole)
        pygame.gfxdraw.aapolygon(self.screen, flag_triangle, self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, flag_triangle, self.COLOR_FLAG)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), border_radius=2)
            self.screen.blit(temp_surf, (int(p['pos'][0] - cam_x - p['size']), int(p['pos'][1] - p['size'])))

        player_screen_x = self.player_pos[0] - cam_x
        player_screen_y = self.player_pos[1]
        player_rect = pygame.Rect(int(player_screen_x), int(player_screen_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        glow_size = self.PLAYER_SIZE * 1.8
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = player_rect.center
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW + (80,), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        time_text = self.font_ui.render(f"TIME: {self.timer:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "LEVEL COMPLETE" if self.player_pos[1] < self.SCREEN_HEIGHT else "GAME OVER"
            end_text = self.font_game_over.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time": self.timer,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Minimalist Platformer")
    
    while not done:
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_RIGHT]:
            movement = 1
        elif keys[pygame.K_LEFT]:
            movement = 2
            
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

        if env.game_over:
            pygame.time.wait(1000)
            obs, info = env.reset()
            done = False
            continue

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    pygame.quit()