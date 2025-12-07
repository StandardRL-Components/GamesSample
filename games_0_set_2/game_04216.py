import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to aim your jump, hold Shift for more power, and press Space to leap. "
        "Reach the green platform at the top!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade hopper. Leap between procedurally generated platforms to reach the top. "
        "Don't fall and watch the clock!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS # Max episode length

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_POWER = 14
        self.HORIZONTAL_POWER = 6
        self.BOOST_MULTIPLIER = 1.4
        self.AIR_DRAG = 0.98
        self.PLAYER_SIZE = (18, 18)

        # Colors
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 60, 60, 50)
        self.COLOR_PLATFORM = (240, 240, 240)
        self.COLOR_GOAL = (60, 255, 120)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)

        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.can_jump = None
        self.last_platform_y = None
        self.platforms = None
        self.goal_platform = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.np_random = None

        # This check is for development and ensures compliance.
        # self.validate_implementation() # Optional: can be uncommented for dev checks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS

        start_platform = self._generate_platforms()
        
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_SIZE[1]]
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        
        self.on_ground = True
        self.can_jump = True
        self.last_platform_y = self.player_pos[1]
        
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def _generate_platforms(self):
        self.platforms = []
        
        # 1. Create the starting platform
        start_platform = pygame.Rect(self.WIDTH / 2 - 50, self.HEIGHT - 20, 100, 20)
        self.platforms.append(start_platform)

        # 2. Create the goal platform
        self.goal_platform = pygame.Rect(self.WIDTH / 2 - 60, 40, 120, 20)
        
        # 3. Generate a guaranteed path to the top
        path = []
        current_pos = np.array(start_platform.midtop, dtype=float)
        
        while current_pos[1] > self.goal_platform.bottom + self.JUMP_POWER * 10:
            jump_height = self.np_random.uniform(self.JUMP_POWER * 0.8, self.JUMP_POWER * 1.5)
            max_horizontal_dist = self.HORIZONTAL_POWER * (jump_height / self.GRAVITY) * 0.8

            dy = -self.np_random.uniform(50, 80)
            dx = self.np_random.uniform(-max_horizontal_dist, max_horizontal_dist)
            
            new_pos = current_pos + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], 50, self.WIDTH - 50)
            
            width = self.np_random.uniform(60, 120)
            plat_rect = pygame.Rect(new_pos[0] - width / 2, new_pos[1], width, 15)
            path.append(plat_rect)
            current_pos = np.array(plat_rect.midtop, dtype=float)
        
        # 4. Add some random decorative platforms
        num_deco_platforms = 15
        for _ in range(num_deco_platforms):
            width = self.np_random.uniform(40, 100)
            x = self.np_random.uniform(0, self.WIDTH - width)
            y = self.np_random.uniform(self.goal_platform.bottom + 20, self.HEIGHT - 50)
            deco_plat = pygame.Rect(x, y, width, 15)
            
            # Avoid placing deco platforms on top of path platforms
            is_overlapping = any(deco_plat.colliderect(p) for p in path)
            if not is_overlapping:
                self.platforms.append(deco_plat)
        
        self.platforms.extend(path)
        return start_platform

    def step(self, action):
        reward = 0
        self.game_over = False

        # --- 1. Handle Input & Actions ---
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.can_jump and space_pressed:
            jump_power = self.JUMP_POWER * (self.BOOST_MULTIPLIER if shift_held else 1.0)
            self.player_vel[1] = -jump_power
            
            if movement == 3: # Left
                self.player_vel[0] = -self.HORIZONTAL_POWER
            elif movement == 4: # Right
                self.player_vel[0] = self.HORIZONTAL_POWER
            else: # Up, None, or Down
                self.player_vel[0] = 0

            self.on_ground = False
            self.can_jump = False
            reward -= 1 # Small penalty for jumping (cost of action)
            # // SFX: Jump

        # --- 2. Update Physics ---
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
            self.player_vel[0] *= self.AIR_DRAG
        
        # Continuous movement rewards
        if self.player_vel[1] < 0: # Moving up
            reward += 0.1
        elif self.player_vel[1] > 0: # Moving down
            reward -= 0.01

        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # --- 3. Boundary Checks ---
        # Keep player horizontally within screen
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
            self.player_vel[0] = 0
        elif self.player_pos[0] > self.WIDTH - self.PLAYER_SIZE[0]:
            self.player_pos[0] = self.WIDTH - self.PLAYER_SIZE[0]
            self.player_vel[0] = 0
            
        self.player_rect.topleft = self.player_pos

        # --- 4. Collision Detection ---
        self.on_ground = False
        # FIX: Changed > to >= to handle the case where the player is stationary on a platform (vel[1] == 0).
        # This prevents the player from falling through the ground during no-op steps.
        if self.player_vel[1] >= 0: # Check for landing or being on ground
            all_platforms = self.platforms + [self.goal_platform]
            for plat in all_platforms:
                if self.player_rect.colliderect(plat):
                    # Check if player was above the platform in the previous frame
                    if self.player_pos[1] + self.PLAYER_SIZE[1] - self.player_vel[1] <= plat.top + 1: # Added tolerance
                        self.player_pos[1] = plat.top - self.PLAYER_SIZE[1]
                        self.player_vel[1] = 0
                        self.player_vel[0] = 0 # Stop horizontal movement on land
                        self.on_ground = True
                        self.can_jump = True
                        self._create_landing_particles(plat.midtop)
                        # // SFX: Land
                        
                        # Reward for landing on a higher platform
                        if plat.y < self.last_platform_y:
                            reward += 1
                        self.last_platform_y = plat.y

                        # Check for win condition
                        if plat == self.goal_platform:
                            self.game_over = True
                            reward += 100
                            # // SFX: Win
                        break
        
        # --- 5. Update Game State ---
        self.steps += 1
        self.time_left -= 1
        self._update_particles()
        
        # --- 6. Check Termination Conditions ---
        terminated = self.game_over
        
        if self.player_pos[1] > self.HEIGHT: # Fell off screen
            terminated = True
            reward -= 100
            # // SFX: Fall
            
        if self.time_left <= 0 and not terminated:
            terminated = True
            reward -= 10 # Penalty for timeout
            # // SFX: Timeout
            
        truncated = self.steps >= self.MAX_STEPS

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_landing_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
        
        # Goal Platform
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_platform, border_radius=3)

        # Particles
        for p in self.particles:
            size = max(0, p['life'] / 5)
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['pos'][0]), int(p['pos'][1])), int(size))
            
        # Player Glow
        glow_rect = self.player_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=4)
        
    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_left // self.FPS):02d}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs = self._get_observation()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    # The environment is created in "headless" mode by default.
    # To render for a human, we need to unset the SDL_VIDEODRIVER.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # This is a basic example of how to control the environment with a keyboard.
    # It's not a perfect mapping but demonstrates the action space.
    
    obs, info = env.reset()
    done = False
    
    # Pygame window for human interaction
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = [0, 0, 0] # Start with a no-op action

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # actions[1]: Space button (0=released, 1=held)
        # actions[2]: Shift button (0=released, 1=held)
        
        action = [0, 0, 0]
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Info: {info}")
            # Optional: Short pause before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            action = [0, 0, 0]
            
    env.close()