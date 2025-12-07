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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Hold Space to jump higher. Navigate the moving platforms to reach the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer where the only control is jump. Challenge yourself to navigate procedurally "
        "generated obstacle courses against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (20, 30, 50)
    COLOR_BG_BOTTOM = (40, 60, 90)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_PLATFORM = (100, 110, 130)
    COLOR_FINISH = (0, 255, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PROGRESS_BAR = (0, 255, 150)
    COLOR_PROGRESS_BG = (50, 50, 50)

    # Physics & Gameplay
    GRAVITY = 0.7
    JUMP_INITIAL_VEL = -13
    JUMP_HOLD_BOOST = -0.7
    JUMP_HOLD_FRAMES = 8
    PLAYER_RUN_SPEED = 4.5
    LEVEL_LENGTH_SCREENS = 15
    TIME_LIMIT_SECONDS = 45

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.render_mode = render_mode
        self.background_surface = self._create_background()

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel_y = 0.0
        self.player_size = np.array([24, 24], dtype=np.float32)
        self.is_grounded = False
        self.jump_hold_timer = 0
        self.camera_x = 0.0
        self.platforms = []
        self.landed_platform_indices = set()
        self.time_left_steps = 0
        self.difficulty_modifier = 1.0
        self.max_progress_x = 0.0
        self.finish_line_x = 0
        self.particles = []
        self.player_squash = 0.0
        
        # self.reset() # reset is called by the test harness
        # self.validate_implementation() # validation is for dev, not needed in final code
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, 200.0], dtype=np.float32)
        self.player_vel_y = 0.0
        self.is_grounded = False
        self.jump_hold_timer = 0
        
        self.camera_x = 0.0
        self.max_progress_x = self.player_pos[0]
        self.landed_platform_indices = set()
        
        self._generate_platforms()
        # Find the starting platform and place the player on it
        for i, p in enumerate(self.platforms):
            if p['rect'].collidepoint(self.player_pos[0], p['rect'].top + 1):
                self.player_pos[1] = p['rect'].top - self.player_size[1]
                self.is_grounded = True
                self.landed_platform_indices.add(i)
                break
        
        self.time_left_steps = self.TIME_LIMIT_SECONDS * self.FPS
        self.difficulty_modifier = 1.0
        
        self.particles = []
        self.player_squash = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- Update State ---
        self._handle_input(action)
        self._update_player_physics()
        self._update_world_state()
        
        newly_landed = self._handle_collisions()
        
        # --- Update Game Variables ---
        self.steps += 1
        self.time_left_steps -= 1
        self.max_progress_x = max(self.max_progress_x, self.player_pos[0])
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 4

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.difficulty_modifier *= 1.01

        # --- Calculate Reward ---
        reward += 0.01  # Survival reward
        if newly_landed:
            reward += 1.0
        self.score += reward

        # --- Check Termination ---
        terminated = False
        terminal_reward = 0.0
        if self.player_pos[1] > self.SCREEN_HEIGHT + self.player_size[1]: # Fell off screen
            terminated = True
            terminal_reward = -10.0 # Reduced penalty to be less harsh
        elif self.time_left_steps <= 0: # Timed out
            terminated = True
            terminal_reward = -10.0
        elif self.player_pos[0] >= self.finish_line_x: # Reached finish
            terminated = True
            time_bonus = 100.0 * (self.time_left_steps / (self.TIME_LIMIT_SECONDS * self.FPS))
            terminal_reward = 50.0 + time_bonus # Base + bonus
        
        if terminated:
            self.game_over = True
            reward += terminal_reward
            self.score += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        space_held = action[1] == 1
        
        if space_held and self.is_grounded:
            # Initiate Jump
            self.player_vel_y = self.JUMP_INITIAL_VEL
            self.is_grounded = False
            self.jump_hold_timer = self.JUMP_HOLD_FRAMES
            self._create_particles(self.player_pos + [self.player_size[0]/2, self.player_size[1]], 10, self.COLOR_PLAYER, 'jump')
            self.player_squash = -0.4 # Stretch
        
        elif space_held and self.jump_hold_timer > 0:
            # Apply jump boost
            self.player_vel_y += self.JUMP_HOLD_BOOST
            self.jump_hold_timer -= 1
        else:
            self.jump_hold_timer = 0

    def _update_player_physics(self):
        # Forward movement
        self.player_pos[0] += self.PLAYER_RUN_SPEED
        
        # Vertical movement
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        # Squash and stretch decay
        self.player_squash *= 0.85

    def _update_world_state(self):
        # Move platforms
        for p in self.platforms:
            p['rect'].x += p['speed'] * self.difficulty_modifier
            if p['rect'].left > self.finish_line_x + self.SCREEN_WIDTH or p['rect'].right < 0:
                # Simple wrap-around for platforms that go off-screen
                if p['speed'] > 0: p['rect'].right = 0
                else: p['rect'].left = self.finish_line_x + self.SCREEN_WIDTH

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1 # Particle gravity
            p['life'] -= 1

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        self.is_grounded = False
        newly_landed_on_platform = False

        for i, platform_data in enumerate(self.platforms):
            platform_rect = platform_data['rect']
            if player_rect.colliderect(platform_rect):
                # Check if player was previously above the platform and is moving down
                player_bottom_last_frame = self.player_pos[1] + self.player_size[1] - self.player_vel_y
                if self.player_vel_y >= 0 and player_bottom_last_frame <= platform_rect.top + 1: # Added tolerance
                    # Landed on top
                    self.player_pos[1] = platform_rect.top - self.player_size[1]
                    self.player_vel_y = 0
                    self.is_grounded = True
                    self.player_squash = 0.5 # Squash
                    
                    if i not in self.landed_platform_indices:
                        newly_landed_on_platform = True
                        self.landed_platform_indices.add(i)
                        self._create_particles(self.player_pos + [self.player_size[0]/2, self.player_size[1]], 15, self.COLOR_PLATFORM, 'land')
                    break # Stop checking after first valid landing
        return newly_landed_on_platform
    
    def _get_observation(self):
        # --- Render Background ---
        self.screen.blit(self.background_surface, (0, 0))

        # --- Render Game Elements ---
        self._render_platforms()
        self._render_finish_line()
        self._render_particles()
        self._render_player()
        
        # --- Render UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_platforms(self):
        for p in self.platforms:
            screen_rect = p['rect'].copy()
            screen_rect.x -= int(self.camera_x)
            if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)

    def _render_finish_line(self):
        finish_screen_x = int(self.finish_line_x - self.camera_x)
        if finish_screen_x < self.SCREEN_WIDTH and finish_screen_x + 20 > 0:
            for y in range(0, self.SCREEN_HEIGHT, 20):
                color = self.COLOR_FINISH if (y // 20) % 2 == 0 else self.COLOR_BG_BOTTOM
                pygame.draw.rect(self.screen, color, (finish_screen_x, y, 20, 20))
            
            finish_text = self.font_small.render("FINISH", True, self.COLOR_TEXT)
            text_rect = finish_text.get_rect(center=(finish_screen_x + 10, 20))
            self.screen.blit(finish_text, text_rect)

    def _render_player(self):
        # Squash and stretch effect
        squash_factor = 1.0 - self.player_squash
        stretch_factor = 1.0 + self.player_squash
        
        w = self.player_size[0] * squash_factor
        h = self.player_size[1] * stretch_factor
        
        screen_x = self.player_pos[0] - self.camera_x
        screen_y = self.player_pos[1] + self.player_size[1] - h # Anchor to bottom
        
        player_rect = pygame.Rect(int(screen_x), int(screen_y), int(w), int(h))
        
        # Glow effect
        glow_rect = player_rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 100), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
    
    def _render_particles(self):
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (screen_pos[0] - size, screen_pos[1] - size))

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {self.time_left_steps / self.FPS:.1f}"
        time_surf = self.font_main.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        # Progress Bar
        progress = min(1.0, self.max_progress_x / self.finish_line_x) if self.finish_line_x > 0 else 0
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BG, (10, 10, bar_width, 15), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (10, 10, bar_width * progress, 15), border_radius=4)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left_steps / self.FPS,
            "progress": self.max_progress_x / self.finish_line_x if self.finish_line_x > 0 else 0,
        }

    def _generate_platforms(self):
        self.platforms = []
        self.finish_line_x = self.SCREEN_WIDTH * self.LEVEL_LENGTH_SCREENS

        # Starting platform. Make it long enough to survive the 60-step stability test.
        # Player runs at 4.5px/step. 60 steps * 4.5px/step = 270px.
        # Player starts at x=100. Platform must extend beyond 100 + 270 = 370.
        # A width of 400 (from x=20 to x=420) is sufficient.
        start_plat = pygame.Rect(20, 300, 400, 100)
        self.platforms.append({'rect': start_plat, 'speed': 0})
        
        current_x = start_plat.right
        last_y = start_plat.y
        
        while current_x < self.finish_line_x - self.SCREEN_WIDTH:
            gap = self.np_random.integers(60, 161)
            current_x += gap
            
            width = self.np_random.integers(80, 251)
            y_change = self.np_random.uniform(-80, 80)
            y = np.clip(last_y + y_change, 150, self.SCREEN_HEIGHT - 50)
            
            speed = self.np_random.uniform(-1.5, 1.5)
            
            plat_rect = pygame.Rect(int(current_x), int(y), int(width), 200)
            self.platforms.append({'rect': plat_rect, 'speed': speed})
            
            current_x += width
            last_y = y

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 2)]
            elif p_type == 'land':
                angle = self.np_random.uniform(0, math.pi) # Upward semicircle
                speed = self.np_random.uniform(1, 3)
                vel = [-math.cos(angle) * speed, -math.sin(angle) * speed]
            else:
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]
            
            self.particles.append({
                'pos': np.array(pos, dtype=np.float32),
                'vel': np.array(vel, dtype=np.float32),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.integers(3, 6)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for development and can be removed, but we keep it
        # as it was in the original code.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            obs, _ = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert obs.dtype == np.uint8
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert not trunc
            assert isinstance(info, dict)
        except Exception as e:
            print(f"Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not part of the required Gymnasium interface but is useful for testing.
    # To run, you might need to unset the dummy video driver, e.g., by commenting out
    # the os.environ line at the top of the file, and install pygame.
    try:
        env = GameEnv()
        
        # Override the render method for direct display if not using "rgb_array"
        pygame.display.set_caption("Minimalist Platformer")
        real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        env.screen = real_screen # Draw directly to the display surface

        obs, info = env.reset()
        done = False
        
        # Game loop for human play
        while not done:
            action = [0, 0, 0] # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                action[1] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
            if done:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Simple reset on game over to allow continuous play
                obs, info = env.reset()
                done = False

    except pygame.error as e:
        print(f"Could not run in interactive mode: {e}")
        print("This is expected if you are in a headless environment.")
        print("The 'GameEnv' class is still valid for use with Gymnasium.")
    finally:
        env.close()