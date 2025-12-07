import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:54:07.116912
# Source Brief: brief_00091.md
# Brief Index: 91
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for an arcade-style spaceship game.
    The player pilots a ship through an asteroid field to collect crystals.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship through a dangerous asteroid field to collect all the energy crystals before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓←→ or WASD to control your ship. ↑/W for forward thrust, "
        "↓/S for reverse, and ←→/AD to rotate."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_SHIP = (255, 255, 255)
    COLOR_THRUST = (255, 180, 50)
    COLOR_CRYSTAL = (100, 200, 255)
    COLOR_CRYSTAL_SPARKLE = (200, 255, 255)
    COLOR_ASTEROID = (200, 80, 80)
    COLOR_ASTEROID_OUTLINE = (150, 50, 50)
    COLOR_UI_TEXT = (50, 255, 50)
    
    # Game Parameters
    SHIP_SIZE = 12
    SHIP_TURN_RATE = 5.0  # degrees per step
    SHIP_THRUST_POWER = 0.15
    SHIP_MAX_SPEED = 5.0
    SHIP_DRAG = 0.99

    CRYSTAL_COUNT = 15
    CRYSTAL_SIZE = 8
    CRYSTAL_COLLECT_RADIUS = 20

    ASTEROID_COUNT = 10
    ASTEROID_MIN_SIZE = 15
    ASTEROID_MAX_SIZE = 35
    ASTEROID_MIN_SPEED = 0.1
    ASTEROID_MAX_SPEED = 0.5
    ASTEROID_MIN_ROT_SPEED = -1.0
    ASTEROID_MAX_ROT_SPEED = 1.0

    MAX_STEPS = 2700  # 45 seconds at 60 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State Variables ---
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_angle = 0.0
        self.thrusting = False
        
        self.asteroids = []
        self.crystals = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Player ---
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_angle = -90.0  # Pointing up
        self.thrusting = False
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False

        # --- Generate Scenery and Collectibles ---
        self._generate_stars()
        self._generate_asteroids()
        self._generate_crystals()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Unpack Action ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        # --- Calculate Reward ---
        reward = self._calculate_reward()

        # --- Update Game Logic ---
        self.steps += 1
        self._handle_player_input(movement)
        self._update_player()
        self._update_asteroids()
        self._handle_collisions()

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.crystals_collected >= self.CRYSTAL_COUNT:
                reward += 50.0  # Win bonus
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Helper Methods for Game Logic ---

    def _handle_player_input(self, movement):
        # Action mapping from brief: 1=rot ccw, 2=rot cw, 3=thrust fwd, 4=thrust back
        if movement == 1:  # Rotate CCW
            self.player_angle -= self.SHIP_TURN_RATE
        if movement == 2:  # Rotate CW
            self.player_angle += self.SHIP_TURN_RATE
        
        self.thrusting = False
        thrust_direction = 0
        if movement == 3:  # Thrust Forward
            self.thrusting = True
            thrust_direction = 1
            # SFX: Thrust sound start
        if movement == 4:  # Thrust Backward
            self.thrusting = True
            thrust_direction = -0.5 # Less power for retro-thrusters
            # SFX: Retro-thrust sound start

        if self.thrusting:
            angle_rad = math.radians(self.player_angle)
            thrust_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.SHIP_THRUST_POWER * thrust_direction
            self.player_vel += thrust_vec

    def _update_player(self):
        # Apply drag
        self.player_vel *= self.SHIP_DRAG
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.SHIP_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.SHIP_MAX_SPEED
            
        # Update position
        self.player_pos += self.player_vel
        
        # World wrap-around
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']
            
            # World wrap-around
            asteroid['pos'][0] %= self.SCREEN_WIDTH
            asteroid['pos'][1] %= self.SCREEN_HEIGHT

    def _handle_collisions(self):
        # Player and Crystals
        for crystal in self.crystals[:]:
            dist = np.linalg.norm(self.player_pos - crystal['pos'])
            if dist < self.CRYSTAL_COLLECT_RADIUS:
                self.crystals.remove(crystal)
                self.score += 5
                self.crystals_collected += 1
                # SFX: Crystal collect sound
                
        # Player and Asteroids
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < asteroid['size'] + self.SHIP_SIZE:
                self.score -= 20
                self.game_over = True # Collision ends the game
                # SFX: Explosion sound
                break

    def _calculate_reward(self):
        # Continuous reward for moving towards the nearest crystal
        if not self.crystals:
            return 0.0

        # Find distance to nearest crystal before move
        dist_before = min(np.linalg.norm(self.player_pos - c['pos']) for c in self.crystals)
        
        # Simulate next position
        next_pos = (self.player_pos + self.player_vel)
        next_pos[0] %= self.SCREEN_WIDTH
        next_pos[1] %= self.SCREEN_HEIGHT
        
        dist_after = min(np.linalg.norm(next_pos - c['pos']) for c in self.crystals)
        
        # Reward is positive if we got closer
        return 0.1 if dist_after < dist_before else -0.05

    def _check_termination(self):
        if self.game_over: # Set by collision
            return True
        if self.crystals_collected >= self.CRYSTAL_COUNT:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    # --- Helper Methods for State Generation ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'size': random.choice([1, 1, 1, 2]),
                'color': random.choice([(100,100,100), (150,150,150), (200,200,200)])
            })

    def _generate_safe_pos(self, min_dist_from_center):
        while True:
            pos = np.array([
                random.uniform(0, self.SCREEN_WIDTH),
                random.uniform(0, self.SCREEN_HEIGHT)
            ], dtype=np.float32)
            if np.linalg.norm(pos - self.player_pos) > min_dist_from_center:
                return pos

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            size = random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
            pos = self._generate_safe_pos(size * 3) # Spawn further away
            
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
            
            num_vertices = random.randint(7, 12)
            points = []
            for i in range(num_vertices):
                angle_vert = (2 * math.pi / num_vertices) * i
                dist = random.uniform(size * 0.7, size)
                points.append((dist * math.cos(angle_vert), dist * math.sin(angle_vert)))

            self.asteroids.append({
                'pos': pos,
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'size': size,
                'angle': 0,
                'rot_speed': random.uniform(self.ASTEROID_MIN_ROT_SPEED, self.ASTEROID_MAX_ROT_SPEED),
                'points': points
            })

    def _generate_crystals(self):
        self.crystals = []
        for _ in range(self.CRYSTAL_COUNT):
            self.crystals.append({
                'pos': self._generate_safe_pos(100),
                'phase': random.uniform(0, 2 * math.pi) # For sparkle animation
            })

    # --- Helper Methods for Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_crystals()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

    def _render_player(self):
        angle_rad = math.radians(self.player_angle)
        x, y = int(self.player_pos[0]), int(self.player_pos[1])

        # Ship body
        p1 = (x + self.SHIP_SIZE * math.cos(angle_rad), y + self.SHIP_SIZE * math.sin(angle_rad))
        p2 = (x + self.SHIP_SIZE * 0.5 * math.cos(angle_rad + math.pi * 0.8), y + self.SHIP_SIZE * 0.5 * math.sin(angle_rad + math.pi * 0.8))
        p3 = (x + self.SHIP_SIZE * 0.5 * math.cos(angle_rad - math.pi * 0.8), y + self.SHIP_SIZE * 0.5 * math.sin(angle_rad - math.pi * 0.8))
        ship_points = [p1, p2, p3]
        
        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_SHIP)
        
        # Thrust flame
        if self.thrusting:
            flame_size = self.SHIP_SIZE * (0.8 + 0.2 * math.sin(self.steps * 0.8))
            p_flame_1 = (x - self.SHIP_SIZE * 0.6 * math.cos(angle_rad), y - self.SHIP_SIZE * 0.6 * math.sin(angle_rad))
            p_flame_2 = (x + flame_size * 0.4 * math.cos(angle_rad + math.pi * 0.9), y + flame_size * 0.4 * math.sin(angle_rad + math.pi * 0.9))
            p_flame_3 = (x + flame_size * 0.4 * math.cos(angle_rad - math.pi * 0.9), y + flame_size * 0.4 * math.sin(angle_rad - math.pi * 0.9))
            flame_points = [p_flame_1, p_flame_2, p_flame_3]
            
            pygame.gfxdraw.aapolygon(self.screen, flame_points, self.COLOR_THRUST)
            pygame.gfxdraw.filled_polygon(self.screen, flame_points, self.COLOR_THRUST)
            
    def _render_crystals(self):
        for crystal in self.crystals:
            x, y = int(crystal['pos'][0]), int(crystal['pos'][1])
            phase = crystal['phase'] + self.steps * 0.1
            size = self.CRYSTAL_SIZE * (1 + 0.15 * math.sin(phase))
            
            points = [
                (x, y - size), (x + size, y),
                (x, y + size), (x - size, y)
            ]
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
            
            # Sparkle effect
            sparkle_size = size * 0.5
            sparkle_points = [
                (x - sparkle_size, y - sparkle_size), (x + sparkle_size, y + sparkle_size),
                (x - sparkle_size, y + sparkle_size), (x + sparkle_size, y - sparkle_size)
            ]
            pygame.draw.line(self.screen, self.COLOR_CRYSTAL_SPARKLE, sparkle_points[0], sparkle_points[1], 1)
            pygame.draw.line(self.screen, self.COLOR_CRYSTAL_SPARKLE, sparkle_points[2], sparkle_points[3], 1)


    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = int(asteroid['pos'][0]), int(asteroid['pos'][1])
            angle = math.radians(asteroid['angle'])
            
            rotated_points = []
            for px, py in asteroid['points']:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                rotated_points.append((x + rx, y + ry))

            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Crystal Counter
        crystal_text = self.font_ui.render(f"CRYSTALS: {self.crystals_collected}/{self.CRYSTAL_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (self.SCREEN_WIDTH/2 - crystal_text.get_width()/2, self.SCREEN_HEIGHT - 30))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        if self.crystals_collected >= self.CRYSTAL_COUNT:
            msg = "MISSION COMPLETE"
        else:
            msg = "GAME OVER"
            
        game_over_text = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
        text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected,
            "time_remaining": (self.MAX_STEPS - self.steps)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
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


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Mapping keyboard keys to actions
    key_map = {
        pygame.K_w: 3,  # Thrust forward
        pygame.K_UP: 3,
        pygame.K_s: 4,  # Thrust backward
        pygame.K_DOWN: 4,
        pygame.K_a: 1,  # Rotate CCW
        pygame.K_LEFT: 1,
        pygame.K_d: 2,  # Rotate CW
        pygame.K_RIGHT: 2,
    }

    # Create a display for manual playing
    pygame.display.set_caption("Spaceship Collector")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while running:
        # --- Create Action from Keyboard Input ---
        action = [0, 0, 0] # Default action: no-op
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key found in map order

        # --- Handle Pygame Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Step the Environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        # So we need to transpose back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Run at 60 FPS for smooth visuals

    env.close()