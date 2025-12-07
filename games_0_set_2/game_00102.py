
# Generated: 2025-08-27T12:36:24.670217
# Source Brief: brief_00102.md
# Brief Index: 102

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move horizontally. Press space to jump from a platform."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Side-view arcade platformer. Control a space hopper, jumping between "
        "procedurally generated platforms to reach the top. Don't fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (255, 64, 64)
    COLOR_PLAYER_INNER = (255, 180, 180)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_GOAL_PLATFORM = (255, 215, 0)
    COLOR_PARTICLE = (220, 220, 220)
    COLOR_TEXT = (255, 255, 255)

    # Physics
    GRAVITY = 0.8
    JUMP_VELOCITY = -14
    PLAYER_SPEED = 5
    PLAYER_RADIUS = 12
    MAX_STEPS = 1500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables are initialized in reset()
        self.platforms = []
        self.hopper_pos = [0, 0]
        self.hopper_vel = [0, 0]
        self.on_platform = False
        self.landings = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def _generate_platforms(self):
        self.platforms.clear()
        
        # Starting platform
        start_platform = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 40, 100, 15)
        self.platforms.append(start_platform)
        
        current_y = start_platform.y
        
        while current_y > 50:
            difficulty_level = min(15, self.landings // 5)
            
            # Decrease platform width with difficulty
            min_width = max(40, 120 - difficulty_level * 5)
            max_width = max(60, 160 - difficulty_level * 5)
            width = self.np_random.integers(min_width, max_width + 1)
            
            # Increase potential gap with difficulty
            max_horiz_gap = 100 + difficulty_level * 10
            max_vert_gap = 60 + difficulty_level * 4
            
            prev_platform = self.platforms[-1]
            
            # Position new platform relative to the previous one
            px = prev_platform.centerx
            dx = self.np_random.integers(-max_horiz_gap, max_horiz_gap + 1)
            x = px + dx - width / 2
            x = np.clip(x, 20, self.WIDTH - width - 20) # Keep platforms on screen
            
            dy = self.np_random.integers(40, max_vert_gap + 1)
            y = current_y - dy
            
            new_platform = pygame.Rect(int(x), int(y), int(width), 15)
            self.platforms.append(new_platform)
            current_y = y
            
        # Goal platform
        goal_x = self.np_random.integers(50, self.WIDTH - 150)
        self.platforms[-1] = pygame.Rect(goal_x, 30, 100, 20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.landings = 0
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.hopper_pos = [start_platform.centerx, start_platform.top - self.PLAYER_RADIUS]
        self.hopper_vel = [0, 0]
        self.on_platform = True
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = 0
        old_pos = list(self.hopper_pos)

        # --- Handle Actions ---
        if movement == 3:  # Left
            self.hopper_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.hopper_pos[0] += self.PLAYER_SPEED
        
        # Clamp player horizontal position
        self.hopper_pos[0] = np.clip(self.hopper_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)

        if self.on_platform and space_pressed:
            # --- JUMP ---
            self.hopper_vel[1] = self.JUMP_VELOCITY
            self.on_platform = False
            # Sound: Jump sfx
            self._create_particles(self.hopper_pos[0], self.hopper_pos[1] + self.PLAYER_RADIUS, 5, (0, -1))

        # --- Physics Update ---
        if not self.on_platform:
            self.hopper_vel[1] += self.GRAVITY
            self.hopper_pos[1] += self.hopper_vel[1]

        # --- Collision Detection ---
        collided_platform = None
        if self.hopper_vel[1] > 0: # Only check for landing if falling
            player_rect = pygame.Rect(
                self.hopper_pos[0] - self.PLAYER_RADIUS, 
                self.hopper_pos[1] - self.PLAYER_RADIUS, 
                self.PLAYER_RADIUS * 2, 
                self.PLAYER_RADIUS * 2
            )
            for platform in self.platforms:
                # Check if player's bottom is intersecting platform top within the last frame
                if (player_rect.colliderect(platform) and 
                    old_pos[1] + self.PLAYER_RADIUS <= platform.top):
                    collided_platform = platform
                    break
        
        if collided_platform:
            self.on_platform = True
            self.hopper_vel[1] = 0
            self.hopper_pos[1] = collided_platform.top - self.PLAYER_RADIUS
            self.landings += 1
            reward += 1.0
            # Sound: Land sfx
            self._create_particles(self.hopper_pos[0], self.hopper_pos[1] + self.PLAYER_RADIUS, 15)
            
            # Regenerate platforms on every 5th landing for new challenges
            if self.landings > 0 and self.landings % 5 == 0:
                self._generate_platforms()


        # --- Reward Calculation ---
        y_change = old_pos[1] - self.hopper_pos[1] # Positive is up
        if y_change > 0:
            reward += y_change * 0.1
        
        x_change = abs(old_pos[0] - self.hopper_pos[0])
        if x_change > 0:
            reward -= x_change * 0.01

        self.score += reward
        
        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.hopper_pos[1] - self.PLAYER_RADIUS > self.HEIGHT: # Fell off bottom
            terminated = True
            self.game_over = True
            self.score -= 10
            # Sound: Failure sfx
        
        if collided_platform is self.platforms[-1]: # Reached goal
            terminated = True
            self.game_over = True
            self.score += 100
            # Sound: Victory sfx

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, x, y, count, direction=None):
        for _ in range(count):
            if direction:
                angle = math.atan2(direction[1], direction[0]) + self.np_random.uniform(-0.5, 0.5)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'radius': radius, 'life': lifetime})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            # Interpolate color from top to bottom
            ratio = y / self.HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))

        # --- Render Game Elements ---
        self._render_game()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (behind player)
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), self.COLOR_PARTICLE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), self.COLOR_PARTICLE)

        # Render platforms
        for i, p in enumerate(self.platforms):
            color = self.COLOR_GOAL_PLATFORM if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, p, border_radius=3)

        # Render player
        player_pos_int = (int(self.hopper_pos[0]), int(self.hopper_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS // 2, self.COLOR_PLAYER_INNER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS // 2, self.COLOR_PLAYER_INNER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        height = max(0, self.HEIGHT - (self.hopper_pos[1] + self.PLAYER_RADIUS))
        height_text = self.font_ui.render(f"Height: {int(height)}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 35))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, self.HEIGHT - self.hopper_pos[1]),
            "landings": self.landings
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set a dummy video driver to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            env.reset()

    # You can also run this with pygame display for visual testing
    # by commenting out the os.environ line and adding this code:
    #
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Space Hopper")
    # clock = pygame.time.Clock()
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
    #
    #     if keys[pygame.K_SPACE]:
    #         action[1] = 1
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']:.2f}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset()
    #
    #     # Display the frame
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     clock.tick(env.FPS)
    #
    # env.close()