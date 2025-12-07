import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T11:17:46.079204
# Source Brief: brief_00981.md
# Brief Index: 981
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide color-coded asteroids to their matching planets. Use time dilation to steer a selected asteroid and avoid mismatched collisions."
    )
    user_guide = (
        "Use arrow keys to move the reticle. Press space to cycle which asteroid to control. Hold shift to slow time and guide the selected asteroid."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_WHITE = (240, 240, 240)
    COLOR_RETICLE = (255, 0, 128)
    PALETTE = [
        (255, 87, 51),   # Fiery Red
        (51, 182, 255),  # Sky Blue
        (87, 255, 51),   # Lime Green
        (255, 235, 59),  # Bright Yellow
        (188, 51, 255),  # Royal Purple
    ]
    COLOR_TRAJECTORY_SUCCESS = (0, 255, 128, 150)
    COLOR_TRAJECTORY_FAIL = (255, 50, 50, 150)
    COLOR_TRAJECTORY_NEUTRAL = (255, 255, 255, 100)

    # Game Mechanics
    RETICLE_SPEED = 15
    ASTEROID_BASE_SPEED = 2.0
    ASTEROID_RADIUS = 15
    PLANET_RADIUS = 30
    TIME_DILATION_FACTOR = 0.1
    RETICLE_INFLUENCE = 0.05

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 50, bold=True)

        # --- Game State Initialization ---
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        self.planets = []
        self.asteroids = []
        self.particles = []
        self.stars = []

        self.reticle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.selected_asteroid_idx = 0
        self.time_dilation_active = False

        # For handling toggle actions
        self.prev_space_state = 0
        self.prev_shift_state = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.level = 1

        self.reticle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.time_dilation_active = False
        self.selected_asteroid_idx = 0
        
        self.prev_space_state = 0
        self.prev_shift_state = 0

        self._create_stars()
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_state
        shift_pressed = shift_held and not self.prev_shift_state
        self.prev_space_state = space_held
        self.prev_shift_state = shift_held

        if shift_pressed:
            self.time_dilation_active = not self.time_dilation_active
            # SFX: Time Dilation On/Off sound

        if self.time_dilation_active:
            reward -= 0.05 # Small penalty for using time dilation

        # --- Update Game Logic ---
        self._move_reticle(movement)

        if space_pressed and self.asteroids:
            self.selected_asteroid_idx = (self.selected_asteroid_idx + 1) % len(self.asteroids)
            # SFX: UI Select sound

        self._update_asteroids()
        self._update_particles()
        
        collision_reward, terminated = self._check_collisions()
        reward += collision_reward

        # --- Check Termination Conditions ---
        if not self.asteroids and not self.game_over:
            reward += 100.0 # Level complete bonus
            self.game_over_message = "LEVEL COMPLETE"
            self.level += 1
            self._setup_level()
        elif terminated:
            self.game_over = True
            # SFX: Game Over sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME LIMIT REACHED"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    # --- Private Helper Methods: Game Logic ---

    def _setup_level(self):
        self.planets.clear()
        self.asteroids.clear()
        self.particles.clear()
        self.selected_asteroid_idx = 0

        num_entities = min(len(self.PALETTE), 3 + (self.level - 1) // 3)
        current_palette = self.PALETTE[:num_entities]
        
        # Place planets
        for i, color in enumerate(current_palette):
            placed = False
            while not placed:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.PLANET_RADIUS + 50, self.SCREEN_WIDTH - self.PLANET_RADIUS - 50),
                    self.np_random.uniform(self.PLANET_RADIUS + 50, self.SCREEN_HEIGHT - self.PLANET_RADIUS - 50)
                )
                if not any(p['pos'].distance_to(pos) < self.PLANET_RADIUS * 3 for p in self.planets):
                    self.planets.append({'pos': pos, 'color': color, 'radius': self.PLANET_RADIUS})
                    placed = True

        # Spawn asteroids
        asteroid_speed = self.ASTEROID_BASE_SPEED + (self.level - 1) * 0.2
        for i, color in enumerate(current_palette):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ASTEROID_RADIUS)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ASTEROID_RADIUS)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.ASTEROID_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.ASTEROID_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))

            target_point = pygame.Vector2(
                self.np_random.uniform(self.SCREEN_WIDTH * 0.25, self.SCREEN_WIDTH * 0.75),
                self.np_random.uniform(self.SCREEN_HEIGHT * 0.25, self.SCREEN_HEIGHT * 0.75)
            )
            vel = (target_point - pos).normalize() * asteroid_speed
            
            self.asteroids.append({
                'pos': pos, 'vel': vel, 'color': color, 'radius': self.ASTEROID_RADIUS,
                'shape': self._create_asteroid_shape(self.ASTEROID_RADIUS)
            })

    def _move_reticle(self, movement):
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.SCREEN_WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.SCREEN_HEIGHT)

    def _update_asteroids(self):
        if not self.asteroids: return
        
        speed_multiplier = self.TIME_DILATION_FACTOR if self.time_dilation_active else 1.0

        for i, asteroid in enumerate(self.asteroids):
            if i == self.selected_asteroid_idx and self.time_dilation_active:
                # Influence selected asteroid's velocity towards the reticle
                direction_to_reticle = (self.reticle_pos - asteroid['pos']).normalize()
                asteroid['vel'] = asteroid['vel'].lerp(direction_to_reticle * asteroid['vel'].length(), self.RETICLE_INFLUENCE)
            
            asteroid['pos'] += asteroid['vel'] * speed_multiplier

    def _check_collisions(self):
        reward = 0
        terminated = False
        
        for i in range(len(self.asteroids) - 1, -1, -1):
            asteroid = self.asteroids[i]
            
            # Planet collision
            for planet in self.planets:
                if asteroid['pos'].distance_to(planet['pos']) < asteroid['radius'] + planet['radius']:
                    if asteroid['color'] == planet['color']:
                        # Success
                        reward += 10.0
                        self.score += 10
                        self._create_particles(asteroid['pos'], asteroid['color'], 50)
                        self.asteroids.pop(i)
                        # SFX: Success chime
                        if self.asteroids: self.selected_asteroid_idx = self.selected_asteroid_idx % len(self.asteroids)
                    else:
                        # Failure
                        reward -= 10.0
                        self.score -= 10
                        terminated = True
                        self.game_over_message = "MISMATCHED COLLISION"
                        self._create_particles(asteroid['pos'], (100, 100, 100), 30)
                        self.asteroids.pop(i)
                        # SFX: Failure explosion
                    return reward, terminated
            
            # Out of bounds
            if not self.screen.get_rect().inflate(100, 100).collidepoint(asteroid['pos']):
                reward -= 10.0
                terminated = True
                self.game_over_message = "ASTEROID LOST"
                self.asteroids.pop(i)
                return reward, terminated
                
        return reward, terminated

    # --- Private Helper Methods: Visuals & Rendering ---

    def _create_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5),
                'brightness': self.np_random.uniform(50, 150)
            })

    def _create_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(5, 9)
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = self.np_random.uniform(radius * 0.7, radius)
            points.append((r * math.cos(angle), r * math.sin(angle)))
        return points

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'color': color,
                'lifespan': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)

    def _render_game(self):
        # Stars
        for star in self.stars:
            brightness = star['brightness'] + math.sin(self.steps * 0.1 + star['pos'].x) * 20
            color = (min(255, int(brightness)), min(255, int(brightness)), min(255, int(brightness)))
            pygame.draw.circle(self.screen, color, star['pos'], star['size'])

        # Planets
        for planet in self.planets:
            pos = (int(planet['pos'].x), int(planet['pos'].y))
            radius = int(planet['radius'])
            color = planet['color']
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Asteroids
        for i, asteroid in enumerate(self.asteroids):
            points = [(p[0] + asteroid['pos'].x, p[1] + asteroid['pos'].y) for p in asteroid['shape']]
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, asteroid['color'])
                pygame.gfxdraw.aapolygon(self.screen, points, asteroid['color'])
            if i == self.selected_asteroid_idx:
                 pygame.gfxdraw.aacircle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), int(asteroid['radius'] * 1.5), (*asteroid['color'], 150))

        # Trajectory line
        if self.time_dilation_active and self.asteroids:
            self._render_trajectory()

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 40))
            color_with_alpha = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color_with_alpha, p['pos'], 2)

        # Reticle
        if self.time_dilation_active:
            x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
            r = 15
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_RETICLE)
            pygame.gfxdraw.aacircle(self.screen, x, y, r - 2, self.COLOR_RETICLE)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - r, y), (x - r//2, y), 1)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x + r, y), (x + r//2, y), 1)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - r), (x, y - r//2), 1)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y + r), (x, y + r//2), 1)

    def _render_trajectory(self):
        if not self.asteroids or self.selected_asteroid_idx >= len(self.asteroids):
            return
        asteroid = self.asteroids[self.selected_asteroid_idx]
        sim_pos = pygame.Vector2(asteroid['pos'])
        sim_vel = pygame.Vector2(asteroid['vel'])

        if (self.reticle_pos - sim_pos).length() > 0:
            direction_to_reticle = (self.reticle_pos - sim_pos).normalize()
            sim_vel = sim_vel.lerp(direction_to_reticle * sim_vel.length(), self.RETICLE_INFLUENCE)
        
        path_points = [sim_pos]
        color = self.COLOR_TRAJECTORY_NEUTRAL
        
        for _ in range(150):
            sim_pos += sim_vel * self.TIME_DILATION_FACTOR
            path_points.append(pygame.Vector2(sim_pos))
            
            for planet in self.planets:
                if sim_pos.distance_to(planet['pos']) < asteroid['radius'] + planet['radius']:
                    color = self.COLOR_TRAJECTORY_SUCCESS if asteroid['color'] == planet['color'] else self.COLOR_TRAJECTORY_FAIL
                    break
            if color != self.COLOR_TRAJECTORY_NEUTRAL:
                break
        
        if len(path_points) > 1:
            # Draw dotted line
            for i in range(0, len(path_points) - 1, 5):
                start = path_points[i]
                end = path_points[min(i + 2, len(path_points) - 1)]
                pygame.draw.line(self.screen, color, start, end, 2)

    def _render_ui(self):
        # Top-left: Asteroids remaining
        text = f"ASTEROIDS: {len(self.asteroids)}"
        surf = self.font_ui.render(text, True, self.COLOR_WHITE)
        self.screen.blit(surf, (10, 10))

        # Top-right: Level
        text = f"LEVEL: {self.level}"
        surf = self.font_ui.render(text, True, self.COLOR_WHITE)
        self.screen.blit(surf, (self.SCREEN_WIDTH - surf.get_width() - 10, 10))
        
        # Bottom-center: Score
        text = f"SCORE: {self.score}"
        surf = self.font_ui.render(text, True, self.COLOR_WHITE)
        self.screen.blit(surf, (self.SCREEN_WIDTH/2 - surf.get_width()/2, self.SCREEN_HEIGHT - surf.get_height() - 10))

        # Game over message
        if self.game_over:
            surf = self.font_msg.render(self.game_over_message, True, self.COLOR_WHITE)
            text_rect = surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Simple background for readability
            bg_rect = text_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect.topleft)
            
            self.screen.blit(surf, text_rect)

    def _validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is not executed by the evaluation environment but is useful for testing.
    # To run, you need to `pip install pygame` and remove the `os.environ` line at the top.
    
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Manual play setup
    pygame.display.set_caption("Cosmic Kaleidoscope")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    # Remove the validation call for interactive play
    try:
        delattr(GameEnv, 'validate_implementation')
    except AttributeError:
        pass

    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    pygame.quit()