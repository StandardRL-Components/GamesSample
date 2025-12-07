
# Generated: 2025-08-27T13:26:14.390234
# Source Brief: brief_00368.md
# Brief Index: 368

        
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
        "Controls: Use arrow keys (↑↓←→) to pilot your ship. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down arcade environment, dodging asteroids to survive for 60 seconds. The asteroid field becomes denser and faster over time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ASTEROID = (220, 80, 80)
        self.COLOR_EXPLOSION = (255, 150, 0)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_STAR = (180, 180, 200)

        # Player
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 5.0

        # Asteroids
        self.INITIAL_ASTEROID_COUNT = 20
        self.ASTEROID_SPEED_INITIAL = 1.0
        self.ASTEROID_SPEED_INCREASE = 0.05
        self.ASTEROID_COUNT_INCREASE = 1
        self.DIFFICULTY_INTERVAL = 10 * self.FPS # every 10 seconds
        self.ASTEROID_SIZE_MIN = 8
        self.ASTEROID_SIZE_MAX = 20

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- Game State Attributes (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.current_asteroid_speed = 0.0
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Optional: uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self.current_asteroid_speed = self.ASTEROID_SPEED_INITIAL
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROID_COUNT)]
        
        self.particles = []
        
        # Generate a static starfield for the episode
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                "size": self.np_random.uniform(0.5, 1.5)
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Update Player Position
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # 2. Update Asteroids
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"] * self.current_asteroid_speed
            # Screen wrapping
            if asteroid["pos"][0] < -asteroid["radius"]: asteroid["pos"][0] = self.WIDTH + asteroid["radius"]
            if asteroid["pos"][0] > self.WIDTH + asteroid["radius"]: asteroid["pos"][0] = -asteroid["radius"]
            if asteroid["pos"][1] < -asteroid["radius"]: asteroid["pos"][1] = self.HEIGHT + asteroid["radius"]
            if asteroid["pos"][1] > self.HEIGHT + asteroid["radius"]: asteroid["pos"][1] = -asteroid["radius"]

        # 3. Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["radius"] += p["expansion_rate"]

        # 4. Difficulty Scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_asteroid_speed += self.ASTEROID_SPEED_INCREASE
            for _ in range(self.ASTEROID_COUNT_INCREASE):
                self.asteroids.append(self._create_asteroid())

        # 5. Collision Detection
        terminated = False
        reward = 0.01  # Small reward for surviving a frame

        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE * 0.8 + asteroid["radius"]: # Using 0.8 for more forgiving hitbox
                self.game_over = True
                terminated = True
                reward = -10.0  # Penalty for collision
                self._create_explosion(self.player_pos)
                # sfx: player_explosion.wav
                break
        
        # 6. Check for Win Condition
        if not terminated and self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward = 100.0  # Large reward for winning
            # sfx: win_jingle.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _create_asteroid(self):
        # Spawn asteroid off-screen
        edge = self.np_random.integers(4)
        radius = self.np_random.uniform(self.ASTEROID_SIZE_MIN, self.ASTEROID_SIZE_MAX)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -radius])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius])
        elif edge == 2: # Left
            pos = np.array([-radius, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT)])
            
        # Aim towards a random point inside the screen to ensure it enters
        target = np.array([self.np_random.uniform(self.WIDTH*0.2, self.WIDTH*0.8), 
                           self.np_random.uniform(self.HEIGHT*0.2, self.HEIGHT*0.8)])
        angle = math.atan2(target[1] - pos[1], target[0] - pos[0])
        vel = np.array([math.cos(angle), math.sin(angle)])
        
        return {"pos": pos, "vel": vel, "radius": radius}

    def _create_explosion(self, pos):
        # sfx: explosion_sound.wav
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 3),
                "expansion_rate": 0, # Not used for this particle type
                "color": self.COLOR_EXPLOSION
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # --- Render all game elements ---
        # 1. Render Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, star["pos"], star["size"])

        # 2. Render Asteroids
        for asteroid in self.asteroids:
            pos_int = (int(asteroid["pos"][0]), int(asteroid["pos"][1]))
            radius_int = int(asteroid["radius"])
            if radius_int > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_ASTEROID)

        # 3. Render Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] > 0:
                alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
                self.screen.blit(temp_surf, p["pos"] - p["radius"])

        # 4. Render Player (only if not game over from collision)
        if not (self.game_over and self.steps < self.MAX_STEPS):
            p1 = (self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE)
            p2 = (self.player_pos[0] - self.PLAYER_SIZE / 1.5, self.player_pos[1] + self.PLAYER_SIZE / 2)
            p3 = (self.player_pos[0] + self.PLAYER_SIZE / 1.5, self.player_pos[1] + self.PLAYER_SIZE / 2)
            points = [p1, p2, p3]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # --- Render UI overlay ---
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"{time_left:.1f}"
        timer_surf = self.font_timer.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (10, 10))
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Asteroid Survival")
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            print("Press 'R' to play again or close the window to quit.")
            
            # Wait for reset or quit
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            print("Resetting game...")
                            obs, info = env.reset()
                            total_reward = 0
                            wait_for_input = False
                        elif event.key == pygame.K_ESCAPE:
                            wait_for_input = False
                            running = False

        # Control the frame rate
        clock.tick(env.FPS)
        
    env.close()