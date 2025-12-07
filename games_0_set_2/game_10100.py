import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:48:14.529075
# Source Brief: brief_00100.md
# Brief Index: 100
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a player navigates a tempest-tossed ship
    through a chaotic sea of glowing, moving geometric tubes.

    The goal is to survive as long as possible. The game's difficulty
    increases both within an episode (tubes move faster) and between
    episodes (the ship's base speed increases upon crashing).

    This environment prioritizes visual quality and "game feel", with smooth
    physics, particle effects, and a high-contrast neon-on-dark aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a ship through a chaotic sea of glowing, moving tubes. "
        "Survive as long as possible as the difficulty increases."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to apply thrust and navigate your ship."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_START = (13, 27, 42)  # Dark stormy blue
    COLOR_BG_END = (0, 0, 0)
    COLOR_SHIP = (255, 255, 255)
    COLOR_THRUSTER = (255, 215, 0) # Gold
    COLOR_UI_TEXT = (255, 255, 0)  # Bright Yellow
    TUBE_COLORS = [
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 69, 0),     # OrangeRed
        (127, 255, 0),    # Chartreuse
    ]

    # Physics & Gameplay
    SHIP_THRUST = 0.5
    SHIP_FRICTION = 0.98
    SHIP_RADIUS = 10
    TUBE_WIDTH = 12
    NUM_TUBES = 12
    PARTICLE_LIFESPAN = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = self.WIDTH
        self.screen_height = self.HEIGHT

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
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        self._create_background()

        # Persistent state (survives resets)
        self.base_ship_speed = 1.0

        # Initialize state variables
        self.ship_pos = None
        self.ship_vel = None
        self.tubes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in production

    def _create_background(self):
        """Creates a pre-rendered background surface for efficiency."""
        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.background, color, (0, y), (self.WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.ship_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.ship_vel = pygame.Vector2(0, 0)
        self.particles = []

        self._spawn_tubes()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _spawn_tubes(self):
        """Initializes the tubes with non-overlapping, safe starting positions."""
        self.tubes.clear()
        padding = 50
        safe_zone = pygame.Rect(
            self.WIDTH/2 - 75, self.HEIGHT/2 - 75, 150, 150
        )

        for _ in range(self.NUM_TUBES):
            while True:
                is_vertical = self.np_random.choice([True, False])
                if is_vertical:
                    x = self.np_random.uniform(padding, self.WIDTH - padding)
                    y = self.np_random.uniform(-self.HEIGHT/2, 0)
                    length = self.np_random.uniform(self.HEIGHT * 0.4, self.HEIGHT * 0.7)
                else:
                    x = self.np_random.uniform(-self.WIDTH/2, 0)
                    y = self.np_random.uniform(padding, self.HEIGHT - padding)
                    length = self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.7)

                # Ensure tubes don't spawn on the player
                temp_rect = pygame.Rect(x, y, self.TUBE_WIDTH if is_vertical else length, length if is_vertical else self.TUBE_WIDTH)
                if not temp_rect.colliderect(safe_zone):
                    break

            self.tubes.append({
                "pos": pygame.Vector2(x, y),
                "is_vertical": is_vertical,
                "length": length,
                "color": random.choice(self.TUBE_COLORS),
                "base_pos": pygame.Vector2(x, y),
                "amplitude": self.np_random.uniform(50, 150),
                "frequency": self.np_random.uniform(0.01, 0.03),
                "phase": self.np_random.uniform(0, 2 * math.pi),
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_ship()
        self._update_tubes()
        self._update_particles()
        
        self.steps += 1
        
        # --- Calculate Reward & Termination ---
        reward = 0.1  # Survival reward
        terminated = False
        truncated = False

        if self._check_collisions():
            reward = -100.0
            terminated = True
            self.game_over = True
            self.base_ship_speed = min(5.0, self.base_ship_speed + 0.2)
        elif self.steps >= self.MAX_STEPS:
            reward += 10.0
            truncated = True # Use truncated for time limits
            self.game_over = True
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        """Applies thrust based on the movement action."""
        thrust_vector = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            thrust_vector.y = -1
        elif movement == 2:  # Down
            thrust_vector.y = 1
        elif movement == 3:  # Left
            thrust_vector.x = -1
        elif movement == 4:  # Right
            thrust_vector.x = 1

        if thrust_vector.length() > 0:
            self.ship_vel += thrust_vector * self.SHIP_THRUST * self.base_ship_speed
            self._spawn_particles(thrust_vector)

    def _spawn_particles(self, thrust_vector):
        """Creates thruster particles."""
        for _ in range(5):
            particle_vel = -thrust_vector * self.np_random.uniform(2, 4) + \
                           pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append({
                "pos": self.ship_pos.copy(),
                "vel": particle_vel,
                "life": self.PARTICLE_LIFESPAN
            })

    def _update_ship(self):
        """Updates ship position and velocity, and handles screen wrap."""
        self.ship_vel *= self.SHIP_FRICTION
        self.ship_pos += self.ship_vel

        # Screen wrapping
        if self.ship_pos.x < 0: self.ship_pos.x = self.WIDTH
        if self.ship_pos.x > self.WIDTH: self.ship_pos.x = 0
        if self.ship_pos.y < 0: self.ship_pos.y = self.HEIGHT
        if self.ship_pos.y > self.HEIGHT: self.ship_pos.y = 0

    def _update_tubes(self):
        """Updates tube positions based on sinusoidal patterns."""
        difficulty_scaling = 1.0 + (self.steps / 100) * 0.05
        difficulty_scaling = min(3.0, difficulty_scaling)

        for tube in self.tubes:
            offset = tube["amplitude"] * math.sin(
                tube["frequency"] * self.steps * difficulty_scaling + tube["phase"]
            )
            if tube["is_vertical"]:
                tube["pos"].x = tube["base_pos"].x + offset
            else:
                tube["pos"].y = tube["base_pos"].y + offset

    def _update_particles(self):
        """Updates particle positions and removes dead ones."""
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _check_collisions(self):
        """Checks for collision between the ship and any tube."""
        for tube in self.tubes:
            tube_rect = pygame.Rect(
                tube["pos"].x,
                tube["pos"].y,
                self.TUBE_WIDTH if tube["is_vertical"] else tube["length"],
                tube["length"] if tube["is_vertical"] else self.TUBE_WIDTH
            )
            # Check collision with the main rect and its wrapped-around counterparts
            for dx in [-self.WIDTH, 0, self.WIDTH]:
                for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                    if tube_rect.move(dx, dy).collidepoint(self.ship_pos):
                        return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.background, (0, 0))

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_tubes()
        self._render_particles()
        self._render_ship()

    def _render_tubes(self):
        for tube in self.tubes:
            rect = pygame.Rect(
                int(tube["pos"].x), int(tube["pos"].y),
                int(self.TUBE_WIDTH) if tube["is_vertical"] else int(tube["length"]),
                int(tube["length"]) if tube["is_vertical"] else int(self.TUBE_WIDTH)
            )
            # Draw for screen wrapping
            for dx in [-self.WIDTH, 0, self.WIDTH]:
                for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                    wrapped_rect = rect.move(dx, dy)
                    self._draw_glowing_rect(self.screen, tube["color"], wrapped_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / self.PARTICLE_LIFESPAN))
            color = (*self.COLOR_THRUSTER, alpha)
            size = int(3 * (p["life"] / self.PARTICLE_LIFESPAN))
            if size > 0:
                # Use a temporary surface for blending to avoid issues with `colorkey`
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p["pos"].x) - size, int(p["pos"].y) - size), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ship(self):
        # Draw glow first
        for i in range(10, 0, -2):
            alpha = 100 - i * 10
            pygame.gfxdraw.aacircle(
                self.screen, int(self.ship_pos.x), int(self.ship_pos.y),
                self.SHIP_RADIUS + i, (*self.COLOR_SHIP, alpha)
            )
        
        # Draw main ship body (as a triangle)
        angle = self.ship_vel.angle_to(pygame.Vector2(1, 0))
        points = []
        for i in range(3):
            point_angle = math.radians(angle + i * 120)
            p_x = self.ship_pos.x + self.SHIP_RADIUS * math.cos(point_angle)
            p_y = self.ship_pos.y - self.SHIP_RADIUS * math.sin(point_angle) # Pygame y is inverted
            points.append((int(p_x), int(p_y)))
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)

    def _render_ui(self):
        time_survived = self.steps / self.FPS
        time_text = self.font.render(f"TIME: {time_survived:.2f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))
        
        speed_text = self.font.render(f"SPEED: x{self.base_ship_speed:.1f}", True, self.COLOR_UI_TEXT)
        speed_rect = speed_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(speed_text, speed_rect)

    def _draw_glowing_rect(self, surface, color, rect):
        """Draws a rectangle with a soft glow effect."""
        glow_color = (*color, 20)
        core_color = color
        
        # Draw glow layers
        for i in range(4, 0, -1):
            glow_rect = rect.inflate(i*2, i*2)
            pygame.draw.rect(surface, glow_color, glow_rect, border_radius=int(self.TUBE_WIDTH/2))
            
        # Draw core rect
        pygame.draw.rect(surface, core_color, rect, border_radius=int(self.TUBE_WIDTH/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ship_speed_multiplier": self.base_ship_speed,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # Set a non-dummy driver for human playback
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    
    # --- Override render method for human play ---
    # This part is for demonstration and not part of the core gym Env
    try:
        env.human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Tempest Tubes")
        
        def render_human(self):
            frame = self._get_observation()
            # The observation is (H, W, C), but pygame wants (W, H) for surface
            # and surfarray.make_surface expects (W, H, C)
            frame_transposed = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_transposed)
            self.human_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.render = render_human.__get__(env, GameEnv)
        # --- End of override ---
        
        obs, info = env.reset()
        done = False
        
        print("\n--- Controls ---")
        print("Arrow Keys: Apply Thrust")
        print("R: Reset Environment")
        print("Q: Quit")
        print("----------------\n")
        
        while not done:
            # Manual control mapping
            movement_action = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            action = [movement_action, 0, 0] # Space and Shift are not used

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"Episode Finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()
                
            # Event handling for quit/reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    if event.key == pygame.K_r:
                        print("Resetting environment...")
                        obs, info = env.reset()

            env.clock.tick(env.FPS)
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Could not set up a display. This might be because you are in a headless environment.")
        print("The environment code is still valid for training in a headless environment.")
    finally:
        env.close()