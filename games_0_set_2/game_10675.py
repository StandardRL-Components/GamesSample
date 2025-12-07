import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:52:57.772400
# Source Brief: brief_00675.md
# Brief Index: 675
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Asteroid Dodger'.
    The player pilots a ship through a horizontally oscillating asteroid field.
    The goal is to survive for a set duration by controlling vertical movement
    and using a limited-duration stealth mode.
    """
    game_description = (
        "Pilot a ship through a dangerous, oscillating asteroid field. "
        "Use your vertical thrusters to navigate and activate a stealth cloak to survive."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to move vertically. "
        "Press and hold space to activate the stealth cloak."
    )
    auto_advance = True

    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_STAR = (100, 100, 120)
    COLOR_SHIP = (255, 255, 255)
    COLOR_SHIP_THRUSTER = (255, 165, 0)
    COLOR_ASTEROID = (180, 180, 190)
    COLOR_UI_TEXT = (0, 255, 150)
    COLOR_STEALTH_BAR = (100, 150, 255)
    COLOR_STEALTH_GLOW = (100, 150, 255, 100) # RGBA for transparency

    # Game Parameters
    MAX_STEPS = 5000
    MAX_COLLISIONS = 15
    INITIAL_ASTEROIDS = 5

    # Player Physics
    PLAYER_X_POS = 100
    PLAYER_ACCELERATION = 20.0  # pixels/sec^2
    PLAYER_MAX_SPEED = 200.0  # pixels/sec
    PLAYER_FRICTION = 0.98
    PLAYER_RADIUS = 10
    PLAYER_INVULNERABILITY_FRAMES = 30 # 1 second

    # Stealth Mechanics
    STEALTH_MAX_ENERGY = 100.0
    STEALTH_DRAIN_RATE = 1.0 # per step
    STEALTH_RECHARGE_RATE = 0.3 # per step
    STEALTH_SPEED_MULTIPLIER = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""

        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_accel_input = 0
        self.player_visual_nudge = 0
        self.invulnerability_timer = 0

        self.stealth_energy = 0.0
        self.stealth_active = False

        self.asteroids = []
        self.base_asteroid_freq = 0.1
        self.num_asteroids = self.INITIAL_ASTEROIDS
        self.stars = []

        self.reset()
        
        # Critical self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.collision_count = 0
        
        self.player_pos = np.array([float(self.PLAYER_X_POS), float(self.SCREEN_HEIGHT / 2)])
        self.player_vel = np.array([0.0, 0.0])
        self.invulnerability_timer = 0

        self.stealth_energy = self.STEALTH_MAX_ENERGY
        self.stealth_active = False

        self.base_asteroid_freq = 0.1
        self.num_asteroids = self.INITIAL_ASTEROIDS
        
        self._generate_stars()
        self._generate_asteroids()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action
        space_held = (space_held == 1)

        self._handle_input(movement, space_held)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated:
            # Apply terminal reward override
            terminal_reward = 0
            if self.collision_count >= self.MAX_COLLISIONS:
                terminal_reward = -100.0
                self.game_over_message = "GAME OVER"
            elif self.steps >= self.MAX_STEPS:
                terminal_reward = 100.0
                self.game_over_message = "SUCCESS"
            
            # Adjust score with terminal reward and return it
            self.score += terminal_reward - reward # subtract step reward to avoid double counting
            reward = terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Reset per-step inputs
        self.player_accel_input = 0
        self.player_visual_nudge = 0

        # Movement
        if movement == 1: # Up
            self.player_accel_input = -1
        elif movement == 2: # Down
            self.player_accel_input = 1
        elif movement == 3: # Left
            self.player_visual_nudge = -2
        elif movement == 4: # Right
            self.player_visual_nudge = 2
            
        # Stealth
        if space_held and self.stealth_energy > 0:
            self.stealth_active = True
        else:
            self.stealth_active = False

    def _update_game_state(self):
        self.steps += 1
        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1

        # Update stealth energy
        if self.stealth_active:
            self.stealth_energy = max(0, self.stealth_energy - self.STEALTH_DRAIN_RATE)
            # SFX: Stealth active hum
        else:
            self.stealth_energy = min(self.STEALTH_MAX_ENERGY, self.stealth_energy + self.STEALTH_RECHARGE_RATE)

        # Update player physics
        accel_per_frame = self.PLAYER_ACCELERATION / self.FPS
        self.player_vel[1] += self.player_accel_input * accel_per_frame
        self.player_vel *= self.PLAYER_FRICTION
        
        max_speed = self.PLAYER_MAX_SPEED / self.FPS
        if self.stealth_active:
            max_speed *= self.STEALTH_SPEED_MULTIPLIER
        
        self.player_vel[1] = np.clip(self.player_vel[1], -max_speed, max_speed)
        self.player_pos += self.player_vel

        # Boundary checks
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Update difficulty
        if self.steps > 0:
            if self.steps % 500 == 0:
                self.base_asteroid_freq += 0.05
            if self.steps % 1000 == 0:
                self.num_asteroids += 1
                self._generate_asteroids()

        # Update asteroids
        time = self.steps / self.FPS
        for asteroid in self.asteroids:
            oscillation = asteroid['amplitude'] * math.sin(
                2 * math.pi * asteroid['freq'] * time + asteroid['phase']
            )
            asteroid['pos'][0] = asteroid['center_x'] + oscillation

    def _calculate_reward(self):
        reward = 0.1  # Survival reward

        # Milestone reward
        if self.steps > 0 and self.steps % 100 == 0:
            reward += 5.0
        
        # Collision penalty
        if self._check_collisions():
            reward -= 10.0
            
        return reward

    def _check_collisions(self):
        if self.invulnerability_timer > 0:
            return False
            
        collided = False
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self.collision_count += 1
                self.invulnerability_timer = self.PLAYER_INVULNERABILITY_FRAMES
                collided = True
                # SFX: Explosion sound
                break # Only one collision per frame
        return collided

    def _check_termination(self):
        if self.collision_count >= self.MAX_COLLISIONS or self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions": self.collision_count,
            "stealth_energy": self.stealth_energy,
        }

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                'depth': random.uniform(0.1, 0.5) # For parallax effect
            })

    def _generate_asteroids(self):
        self.asteroids = []
        for i in range(self.num_asteroids):
            y_pos = (i + 1) * (self.SCREEN_HEIGHT / (self.num_asteroids + 1))
            self.asteroids.append({
                'pos': np.array([0.0, 0.0]), # x is calculated each frame
                'center_x': self.SCREEN_WIDTH / 2,
                'y': y_pos,
                'radius': random.uniform(10, 25),
                'amplitude': random.uniform(self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH * 0.45),
                'freq': self.base_asteroid_freq + random.uniform(-0.05, 0.05),
                'phase': random.uniform(0, 2 * math.pi)
            })
            self.asteroids[-1]['pos'][1] = self.asteroids[-1]['y']


    def _render_game(self):
        self._render_stars()
        self._render_asteroids()
        self._render_player()

    def _render_stars(self):
        for star in self.stars:
            # Simple parallax: stars move slightly opposite to player's logical position
            # Since player is fixed, we can just make them drift slowly
            star['pos'][0] -= star['depth']
            if star['pos'][0] < 0:
                star['pos'][0] = self.SCREEN_WIDTH
                star['pos'][1] = random.uniform(0, self.SCREEN_HEIGHT)
            
            size = int(star['depth'] * 3)
            pygame.draw.rect(self.screen, self.COLOR_STAR, (int(star['pos'][0]), int(star['pos'][1]), size, size))

    def _render_player(self):
        # Determine player color based on state
        player_color = self.COLOR_SHIP
        if self.invulnerability_timer > 0 and self.steps % 10 < 5:
            return # Flashing effect on hit

        # Ship body
        px, py = int(self.player_pos[0] + self.player_visual_nudge), int(self.player_pos[1])
        p1 = (px + self.PLAYER_RADIUS, py)
        p2 = (px - self.PLAYER_RADIUS, py - self.PLAYER_RADIUS // 2)
        p3 = (px - self.PLAYER_RADIUS, py + self.PLAYER_RADIUS // 2)
        points = [p1, p2, p3]
        
        # Stealth glow effect
        if self.stealth_active:
            glow_surface = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, self.COLOR_STEALTH_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS + 5)
            self.screen.blit(glow_surface, (px - self.PLAYER_RADIUS * 2, py - self.PLAYER_RADIUS * 2))

        pygame.gfxdraw.aapolygon(self.screen, points, player_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, player_color)

        # Thruster effect
        if self.player_accel_input != 0:
            # SFX: Thruster sound
            thruster_length = random.randint(5, 15)
            thruster_width = self.PLAYER_RADIUS
            if self.player_accel_input > 0: # Down
                tp1 = (px - self.PLAYER_RADIUS, py - self.PLAYER_RADIUS // 2)
                tp2 = (px - self.PLAYER_RADIUS - thruster_length, py)
                tp3 = (px - self.PLAYER_RADIUS, py + self.PLAYER_RADIUS // 2)
            else: # Up
                tp1 = (px - self.PLAYER_RADIUS, py - self.PLAYER_RADIUS // 2)
                tp2 = (px - self.PLAYER_RADIUS - thruster_length, py)
                tp3 = (px - self.PLAYER_RADIUS, py + self.PLAYER_RADIUS // 2)
                # This is incorrect logic for up/down thrusters on a side-facing ship
                # For a visually correct thruster, it should always be behind the ship.
                # Let's place it at the back regardless of direction.
            thruster_point = (px - self.PLAYER_RADIUS - thruster_length, py)
            thruster_base1 = (px - self.PLAYER_RADIUS, py - thruster_width // 2)
            thruster_base2 = (px - self.PLAYER_RADIUS, py + thruster_width // 2)
            thruster_points = [thruster_base1, thruster_base2, thruster_point]
            pygame.gfxdraw.aapolygon(self.screen, thruster_points, self.COLOR_SHIP_THRUSTER)
            pygame.gfxdraw.filled_polygon(self.screen, thruster_points, self.COLOR_SHIP_THRUSTER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            radius = int(asteroid['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _render_ui(self):
        # Collision Counter
        collision_text = self.font_ui.render(f"Collisions: {self.collision_count}/{self.MAX_COLLISIONS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(collision_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Stealth Bar
        self._render_stealth_bar()

        # Game Over Message
        if self.game_over:
            message_surface = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            rect = message_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(message_surface, rect)

    def _render_stealth_bar(self):
        bar_width = 150
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10

        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # Foreground
        current_width = int(bar_width * (self.stealth_energy / self.STEALTH_MAX_ENERGY))
        pygame.draw.rect(self.screen, self.COLOR_STEALTH_BAR, (bar_x, bar_y, current_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Text label
        stealth_text = self.font_ui.render("Stealth", True, self.COLOR_UI_TEXT)
        self.screen.blit(stealth_text, (bar_x - stealth_text.get_width() - 5, bar_y))

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
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You must unset the dummy videodriver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Dodger")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Special actions
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()