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



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to run, and ↑ to jump. Your goal is to catch the red thief!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced side-scrolling pursuit game. As the blue guard, you must chase down and capture the red thief before they escape or time runs out. The thief gets faster over time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 1280

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Headless execution for verification
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (60, 40, 40)
        self.COLOR_GUARD = (50, 150, 255)
        self.COLOR_THIEF = (255, 80, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE_CAPTURE = [(255, 255, 100), (255, 200, 50)]
        self.COLOR_PARTICLE_LAND = (150, 120, 100)
        
        # Game constants
        self.FPS = 30
        self.GROUND_Y = self.HEIGHT - 60
        self.GRAVITY = 0.8
        self.GUARD_SPEED = 5
        self.JUMP_STRENGTH = 15
        self.MAX_STEPS = 1000
        self.INITIAL_TIMER = 30.0

        # Character dimensions
        self.GUARD_WIDTH, self.GUARD_HEIGHT = 24, 48
        self.THIEF_WIDTH, self.THIEF_HEIGHT = 20, 40
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.game_over = False
        self.guard_pos = np.zeros(2)
        self.guard_vel = np.zeros(2)
        self.thief_pos = np.zeros(2)
        self.thief_speed = 0.0
        self.last_distance_to_thief = 0.0
        self.camera_x = 0.0
        self.particles = []
        self.capture_message = ""
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.INITIAL_TIMER
        self.capture_message = ""

        # Player state
        self.guard_pos = np.array([100.0, self.GROUND_Y])
        self.guard_vel = np.array([0.0, 0.0])

        # Thief state
        self.thief_pos = np.array([300.0, self.GROUND_Y])
        self.thief_speed = 2.0 # Adjusted for better initial gameplay feel

        self.last_distance_to_thief = np.linalg.norm(self.guard_pos - self.thief_pos)
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        # --- UPDATE GAME LOGIC ---
        movement = action[0]
        self._update_guard(movement)
        self._update_thief()
        self._update_particles()

        self.steps += 1
        self.timer -= 1.0 / self.FPS

        # --- CALCULATE REWARD AND CHECK TERMINATION ---
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward

        if terminated:
            self.game_over = True
            # sfx: game_over_sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_guard(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.guard_pos[0] -= self.GUARD_SPEED
        elif movement == 4:  # Right
            self.guard_pos[0] += self.GUARD_SPEED

        # Jumping
        is_on_ground = self.guard_pos[1] >= self.GROUND_Y
        if movement == 1 and is_on_ground:
            self.guard_vel[1] = -self.JUMP_STRENGTH
            # sfx: jump_sound

        # Apply gravity
        self.guard_vel[1] += self.GRAVITY
        self.guard_pos[1] += self.guard_vel[1]

        # Ground collision
        if self.guard_pos[1] > self.GROUND_Y:
            if self.guard_vel[1] > self.GRAVITY * 2: # Landed with some force
                self._create_particles(self.guard_pos[0], self.guard_pos[1] + self.GUARD_HEIGHT / 2, 5, self.COLOR_PARTICLE_LAND)
                # sfx: land_sound
            self.guard_pos[1] = self.GROUND_Y
            self.guard_vel[1] = 0

        # World bounds
        self.guard_pos[0] = np.clip(self.guard_pos[0], self.GUARD_WIDTH / 2, self.WORLD_WIDTH - self.GUARD_WIDTH / 2)

    def _update_thief(self):
        # Thief runs right
        self.thief_pos[0] += self.thief_speed
        
        # Increase speed over time
        if self.steps > 0 and self.steps % 100 == 0:
            self.thief_speed += 0.1 # Increased for more challenge

    def _calculate_reward_and_termination(self):
        reward = 0.0
        terminated = False

        # Reward for closing distance
        current_distance = np.linalg.norm(self.guard_pos - self.thief_pos)
        if current_distance < self.last_distance_to_thief:
            reward += 0.1
        else:
            reward -= 0.1 # Small penalty for moving away
        self.last_distance_to_thief = current_distance

        # Check for capture
        guard_rect = pygame.Rect(self.guard_pos[0] - self.GUARD_WIDTH / 2, self.guard_pos[1] - self.GUARD_HEIGHT, self.GUARD_WIDTH, self.GUARD_HEIGHT)
        thief_rect = pygame.Rect(self.thief_pos[0] - self.THIEF_WIDTH / 2, self.thief_pos[1] - self.THIEF_HEIGHT, self.THIEF_WIDTH, self.THIEF_HEIGHT)
        
        if guard_rect.colliderect(thief_rect):
            capture_reward = 10.0 + 10.0 * max(0, self.timer)
            reward += capture_reward
            terminated = True
            self.capture_message = "CAUGHT!"
            self._create_particles(self.thief_pos[0], self.thief_pos[1] - self.THIEF_HEIGHT / 2, 50, self.COLOR_PARTICLE_CAPTURE)
            # sfx: capture_success_sound
            return reward, terminated

        # Check for thief escape
        if self.thief_pos[0] >= self.WORLD_WIDTH - self.THIEF_WIDTH / 2:
            reward -= 10.0 # Penalty for letting thief escape
            terminated = True
            self.capture_message = "ESCAPED!"
            # sfx: capture_fail_sound
            return reward, terminated

        # Check for timeout or max steps
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.capture_message = "TIME UP!" if self.timer <= 0 else "OUT OF STEPS"
            # sfx: time_up_sound
            return reward, terminated
            
        return reward, terminated

    def _get_observation(self):
        # Update camera to keep both actors in view
        center_point = (self.guard_pos[0] + self.thief_pos[0]) / 2
        self.camera_x = center_point - self.WIDTH / 2
        self.camera_x = np.clip(self.camera_x, 0, self.WORLD_WIDTH - self.WIDTH)

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_ground()
        self._render_thief()
        self._render_guard()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        # Parallax background buildings
        for i in range(5):
            # Far layer
            bx = (100 + i * 250 - self.camera_x * 0.25) % (self.WIDTH + 200) - 100
            bh = 100 + (i % 3) * 30
            pygame.draw.rect(self.screen, (30, 35, 50), (bx, self.GROUND_Y - bh, 150, bh))
            # Mid layer
            bx = (50 + i * 300 - self.camera_x * 0.5) % (self.WIDTH + 150) - 75
            bh = 150 + (i % 2) * 40
            pygame.draw.rect(self.screen, (40, 45, 60), (bx, self.GROUND_Y - bh, 100, bh))

    def _render_ground(self):
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

    def _render_thief(self):
        x, y = self.thief_pos
        w, h = self.THIEF_WIDTH, self.THIEF_HEIGHT
        screen_x = int(x - self.camera_x)
        
        # Simple run animation
        bob = math.sin(self.steps * 0.5) * 2
        
        body_rect = pygame.Rect(screen_x - w/2, y - h + bob, w, h)
        head_rect = pygame.Rect(screen_x - w/3, y - h - 5 + bob, w * 2/3, 10)
        
        pygame.draw.rect(self.screen, self.COLOR_THIEF, body_rect)
        pygame.draw.rect(self.screen, self.COLOR_THIEF, head_rect)

    def _render_guard(self):
        x, y = self.guard_pos
        w, h = self.GUARD_WIDTH, self.GUARD_HEIGHT
        screen_x = int(x - self.camera_x)

        # Run/Jump animation
        bob = 0
        is_on_ground = y >= self.GROUND_Y
        if is_on_ground:
            bob = abs(math.sin(self.steps * 0.4)) * 3
        
        body_rect = pygame.Rect(screen_x - w/2, y - h + bob, w, h)
        head_rect = pygame.Rect(screen_x - w/3, y - h - 6 + bob, w * 2/3, 12)
        
        pygame.draw.rect(self.screen, self.COLOR_GUARD, body_rect)
        pygame.draw.rect(self.screen, self.COLOR_GUARD, head_rect)
        # Glow effect
        pygame.gfxdraw.rectangle(self.screen, body_rect, (*self.COLOR_GUARD, 50))

    def _render_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            color = p['color']
            if isinstance(color, list): # Handle color lists for variety
                color = color[p['life'] % len(color)]
            
            size = max(0, p['size'] * (p['life'] / p['max_life']))
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            pygame.draw.circle(self.screen, color, pos, int(size))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': np.array([x, y]),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 6),
                'color': color
            })

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_color = self.COLOR_TEXT if self.timer > 5 else self.COLOR_THIEF
        timer_text = self.font_small.render(f"TIME: {max(0, self.timer):.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over and self.capture_message:
            msg_text = self.font_large.render(self.capture_message, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "distance_to_thief": self.last_distance_to_thief
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    env = GameEnv()
    env.validate_implementation() # Run the validation check

    # --- To visualize the game, comment out the dummy driver line and run this block ---
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Guard vs. Thief")
    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()

    # while not done:
    #     action = [0, 0, 0] # Default no-op action
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
        
    #     if keys[pygame.K_UP]:
    #         action[0] = 1

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Draw the observation from the environment to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     clock.tick(env.FPS)
    #     if done:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         pygame.time.wait(2000) # Pause before quitting
    
    # env.close()