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
        "Controls: ↑ for a high jump, ↓ for a short hop, ←→ for angled jumps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade platformer. Hop between procedurally generated platforms to reach the top before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    MAX_EPISODE_STEPS = 1800 # 60 seconds * 30 FPS

    # Colors
    COLOR_BG_BOTTOM = (10, 0, 30)
    COLOR_BG_TOP = (60, 10, 80)
    COLOR_PLAYER = (50, 255, 255)
    COLOR_PLAYER_GLOW = (50, 255, 255, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    
    # Player Physics
    PLAYER_SIZE = 12
    GRAVITY = 0.4
    JUMP_FORCE_VERTICAL = -10.5
    JUMP_FORCE_SHORT_HOP = -6.0
    JUMP_FORCE_HORIZONTAL = 6.0
    PLAYER_MAX_VEL_Y = 12
    PLAYER_FRICTION = 0.85

    # Game Mechanics
    INITIAL_TIME = 60.0
    TARGET_HEIGHT = 5000
    PLATFORM_HEIGHT = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.highest_y = None
        self.on_ground = None
        self.can_jump = None
        self.last_platform_y = None
        self.win_platform = None
        
        # Used for smooth background gradient rendering
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self._render_background_gradient()

        # self.reset() is called by the API, no need to call it here.
        # But for validation and standalone use, it's often called.
        # Let's ensure it's safe.
        
        # self.validate_implementation() # Cannot validate before first reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.INITIAL_TIME
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        
        self.particles = []
        self.platforms = []
        
        start_platform = pygame.Rect(
            self.player_pos[0] - 40, self.player_pos[1] + self.PLAYER_SIZE, 80, self.PLATFORM_HEIGHT
        )
        self.platforms.append(start_platform)
        self.last_platform_y = start_platform.y

        self.highest_y = self.player_pos[1]
        self.on_ground = True
        self.can_jump = True

        # FIX: Create the final winning platform BEFORE generating initial platforms
        # that depend on its position.
        self.win_platform = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 100, 
            start_platform.y - self.TARGET_HEIGHT, 
            200, 
            self.PLATFORM_HEIGHT
        )

        # Generate initial platforms
        self._generate_initial_platforms()
        
        # Now add the win platform to the list of platforms to be rendered/collided with.
        self.platforms.append(self.win_platform)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            movement = action[0]
            
            # --- Update Game Logic ---
            self.steps += 1
            self.time_remaining -= 1.0 / self.TARGET_FPS
            reward += 0.01 # Small survival reward

            self._handle_input(movement)
            self._update_player()
            reward += self._check_collisions()
            self._update_platforms()
            self._update_particles()

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.player_pos[1] > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
                reward = -10.0 # Fell off screen
            elif self.time_remaining <= 0:
                reward = -10.0 # Timed out
            elif self._is_on_win_platform():
                reward = 100.0 # Reached the top
            
        self.score += reward

        # `truncated` is always False as per problem description.
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Gymnasium standard: if truncated, terminated should also be true

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if self.can_jump:
            jump_executed = False
            # Action 1: Up Jump
            if movement == 1:
                self.player_vel[1] = self.JUMP_FORCE_VERTICAL
                jump_executed = True
            # Action 2: Short Hop
            elif movement == 2:
                self.player_vel[1] = self.JUMP_FORCE_SHORT_HOP
                jump_executed = True
            # Action 3: Left Jump
            elif movement == 3:
                self.player_vel[1] = self.JUMP_FORCE_VERTICAL * 0.85
                self.player_vel[0] = -self.JUMP_FORCE_HORIZONTAL
                jump_executed = True
            # Action 4: Right Jump
            elif movement == 4:
                self.player_vel[1] = self.JUMP_FORCE_VERTICAL * 0.85
                self.player_vel[0] = self.JUMP_FORCE_HORIZONTAL
                jump_executed = True
            
            if jump_executed:
                self.on_ground = False
                self.can_jump = False
                # sfx: jump_sound()

    def _update_player(self):
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
            self.player_vel[1] = min(self.player_vel[1], self.PLAYER_MAX_VEL_Y)

        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # Screen bounds for horizontal movement
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
            self.player_vel[0] = 0
        elif self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_SIZE:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_SIZE
            self.player_vel[0] = 0

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        if self.player_vel[1] < 0: # Moving up, so can't land
            return reward

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if player was above the platform in the previous frame
                prev_player_bottom = self.player_pos[1] + self.PLAYER_SIZE - self.player_vel[1]
                if prev_player_bottom <= plat.top:
                    # Landed on platform
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.on_ground = True
                    self.can_jump = True
                    # sfx: land_sound()
                    
                    if plat.y < self.last_platform_y:
                        reward += 1.0
                    else:
                        reward += -0.5

                    self.last_platform_y = plat.y
                    self._create_particles(plat.midtop, self._get_platform_color(plat.y), 20)
                    break # Stop checking after first landing
        return reward

    def _update_platforms(self):
        # Update player's highest point (lower y is higher)
        self.highest_y = min(self.highest_y, self.player_pos[1])
        
        # Remove platforms that are off-screen below
        self.platforms = [p for p in self.platforms if p.bottom > self.player_pos[1] - self.SCREEN_HEIGHT * 1.5]

        # Generate new platforms above the screen view if needed
        last_gen_y = min(p.y for p in self.platforms) if self.platforms else self.highest_y
        if last_gen_y > self.highest_y - self.SCREEN_HEIGHT:
            self._generate_platforms(last_gen_y)

    def _generate_initial_platforms(self):
        y = self.platforms[0].y
        for _ in range(20):
            y = self._generate_single_platform(y)
            if y is None: # Stop if we've reached the top
                break

    def _generate_platforms(self, current_top_y):
        y = current_top_y
        for _ in range(5):
            y = self._generate_single_platform(y)
            if y is None: # Stop if we've reached the top
                break
    
    def _generate_single_platform(self, current_y):
        # Difficulty scaling
        difficulty_factor = self.steps / self.MAX_EPISODE_STEPS # 0 to 1
        
        min_width = 40
        max_width = 120
        width = max(min_width, max_width - difficulty_factor * (max_width - min_width))
        
        min_dy, max_dy = 40, 90
        dy = self.np_random.uniform(min_dy, max_dy)
        
        min_dx, max_dx = 80, 150
        dx = self.np_random.uniform(min_dx, max_dx) * self.np_random.choice([-1, 1])

        last_plat_centerx = self.platforms[-1].centerx
        new_x = last_plat_centerx + dx
        new_x = np.clip(new_x, 0, self.SCREEN_WIDTH - width)
        new_y = current_y - dy
        
        if self.win_platform and new_y <= (self.win_platform.y + self.PLATFORM_HEIGHT): # Don't generate past win platform
            return None

        new_platform = pygame.Rect(new_x, new_y, width, self.PLATFORM_HEIGHT)
        self.platforms.append(new_platform)

        return new_y

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] -= 0.1

    def _check_termination(self):
        return (self.player_pos[1] > self.SCREEN_HEIGHT + self.PLAYER_SIZE or
                self.time_remaining <= 0 or
                self._is_on_win_platform())
    
    def _is_on_win_platform(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        return self.on_ground and self.win_platform and player_rect.colliderect(self.win_platform)

    def _get_observation(self):
        # --- Camera Translation ---
        # Keep player in the bottom half of the screen vertically
        camera_offset_y = self.SCREEN_HEIGHT * 0.7 - self.player_pos[1]

        # --- Rendering ---
        # Background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Particles
        self._render_particles(camera_offset_y)
        
        # Platforms
        self._render_platforms(camera_offset_y)

        # Player
        self._render_player(camera_offset_y)
        
        # UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_gradient(self):
        for y in range(self.SCREEN_HEIGHT):
            t = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_BOTTOM[0] * (1 - t) + self.COLOR_BG_TOP[0] * t),
                int(self.COLOR_BG_BOTTOM[1] * (1 - t) + self.COLOR_BG_TOP[1] * t),
                int(self.COLOR_BG_BOTTOM[2] * (1 - t) + self.COLOR_BG_TOP[2] * t),
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_particles(self, cam_y):
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1] + cam_y))
                radius = int(p['radius'])
                alpha = int(max(0, min(255, p['life'] * 5)))
                color = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_platforms(self, cam_y):
        for plat in self.platforms:
            color = self._get_platform_color(plat.y)
            if plat is self.win_platform:
                # Make win platform pulse
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                color = (int(255 * pulse), 255, int(255 * pulse))

            rect = plat.move(0, cam_y)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
    
    def _render_player(self, cam_y):
        player_screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] + cam_y))
        
        # Glow effect
        glow_size = self.PLAYER_SIZE + 10
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = player_screen_pos
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size), border_radius=int(glow_size/2))
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Player square
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = player_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Calculate displayed height
        start_y = self.SCREEN_HEIGHT - 50
        height = max(0, int(start_y - self.player_pos[1]))
        
        # Height display
        height_text = self.font_ui.render(f"HEIGHT: {height}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        # Time display
        time_text = self.font_ui.render(f"TIME: {max(0, self.time_remaining):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score display
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        if self.game_over:
            msg = ""
            if self._is_on_win_platform():
                msg = "YOU WIN!"
            elif self.time_remaining <= 0:
                msg = "TIME UP"
            else:
                msg = "GAME OVER"
            
            over_text = self.font_game_over.render(msg, True, (255, 255, 50))
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, int((self.SCREEN_HEIGHT - 50) - self.player_pos[1])),
            "time_remaining": self.time_remaining
        }
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def _get_platform_color(self, y_pos):
        start_y = self.SCREEN_HEIGHT - 50
        # Interpolate from blue (low) to red (high)
        # We use a non-linear interpolation to get more vibrant mid-tones
        t = max(0, min(1, (start_y - y_pos) / self.TARGET_HEIGHT))
        
        # Blue -> Cyan -> Green -> Yellow -> Red
        if t < 0.25: # Blue to Cyan
            p = t / 0.25
            return (0, int(255 * p), 255)
        elif t < 0.5: # Cyan to Green
            p = (t - 0.25) / 0.25
            return (0, 255, int(255 * (1 - p)))
        elif t < 0.75: # Green to Yellow
            p = (t - 0.5) / 0.25
            return (int(255 * p), 255, 0)
        else: # Yellow to Red
            p = (t - 0.75) / 0.25
            return (255, int(255 * (1 - p)), 0)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # --- Headless execution test ---
    print("--- Running Headless Test ---")
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Run for a few steps to test
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Score={info['score']:.2f}, Terminated={terminated}, Truncated={truncated}")
        if terminated:
            print("Episode finished.")
            obs, info = env.reset(seed=42)
    
    env.close()
    print("--- Headless Test Finished ---\n")
    
    # --- Human play example (requires a display) ---
    try:
        # Remove the dummy driver to allow display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        print("--- Running Human Play Test (requires display) ---")
        print("Controls: Arrow keys UP, DOWN, LEFT, RIGHT. Close window to quit.")
        
        env = GameEnv(render_mode='rgb_array')
        obs, info = env.reset(seed=123)
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Hopper")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # no-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            action = [movement, 0, 0]

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000)
                obs, info = env.reset(seed=124) # Use a different seed for a new level

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(GameEnv.TARGET_FPS)
            
        env.close()
        print("--- Human Play Test Finished ---")
    except Exception as e:
        print("\nCould not run human player example, likely because no display is available.")
        print(f"Error: {e}")