import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:35:46.780629
# Source Brief: brief_00553.md
# Brief Index: 553
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import itertools

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = random.randint(15, 30)  # Frames

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            radius = int(self.lifetime / 5)
            if radius > 0:
                pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), radius)

class Shape:
    """Base class for all geometric shapes in the game."""
    def __init__(self, x, y, color, area):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.area = area
        self.pulse_timer = random.uniform(0, math.pi * 2)
        self.pulse_magnitude = 1.05
        self.creation_tick = 0 # Used for spawn animation
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.type = "shape"

    def update(self, dt=1/30):
        self.pulse_timer += 5 * dt
        self.creation_tick = min(1.0, self.creation_tick + 0.1)

    def get_pulse_scale(self):
        return 1.0 + (math.sin(self.pulse_timer) + 1) / 2 * (self.pulse_magnitude - 1.0)
    
    def get_spawn_scale(self):
        # Ease-out cubic for a snappy spawn effect
        x = 1.0 - self.creation_tick
        return 1.0 - x**3

    def draw(self, surface):
        raise NotImplementedError

class PlayerSquare(Shape):
    """A player-controlled square."""
    def __init__(self, x, y, color, multiplier):
        self.size = 30
        super().__init__(x, y, color, self.size * self.size)
        self.multiplier = multiplier
        self.type = "square"
        self.rect = pygame.Rect(self.pos.x - self.size/2, self.pos.y - self.size/2, self.size, self.size)

    def move(self, dx, dy, bounds):
        self.pos.x += dx
        self.pos.y += dy
        self.pos.x = np.clip(self.pos.x, self.size / 2, bounds.width - self.size / 2)
        self.pos.y = np.clip(self.pos.y, self.size / 2, bounds.height - self.size / 2)
        self.rect.center = self.pos

    def draw(self, surface):
        scale = self.get_pulse_scale() * self.get_spawn_scale()
        size = int(self.size * scale)
        if size <= 0: return
        
        # Glow effect
        glow_color = tuple(min(255, c + 50) for c in self.color)
        for i in range(size // 4, 0, -2):
            alpha = 80 * (1 - (i / (size // 4)))
            temp_color = glow_color + (int(alpha),)
            temp_surf = pygame.Surface((size + i*2, size + i*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, temp_color, temp_surf.get_rect(), border_radius=max(1, int(size*0.2)))
            surface.blit(temp_surf, (int(self.pos.x - size/2 - i), int(self.pos.y - size/2 - i)))

        r = pygame.Rect(int(self.pos.x - size/2), int(self.pos.y - size/2), size, size)
        pygame.draw.rect(surface, self.color, r, border_radius=max(1, int(size*0.2)))

class Circle(Shape):
    """A circle created from a collision."""
    def __init__(self, x, y, color, area):
        super().__init__(x, y, color, area)
        self.radius = math.sqrt(self.area / math.pi)
        self.type = "circle"
        self.rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)

    def draw(self, surface):
        scale = self.get_pulse_scale() * self.get_spawn_scale()
        radius = int(self.radius * scale)
        if radius <= 0: return
        
        # Use gfxdraw for anti-aliased circles
        x, y = int(self.pos.x), int(self.pos.y)
        glow_color = tuple(min(255, c + 50) for c in self.color)
        
        # Glow effect
        for i in range(radius // 4, 0, -1):
            alpha = int(60 * (1 - (i / (radius // 4))))
            if alpha > 0:
                pygame.gfxdraw.aacircle(surface, x, y, radius + i, glow_color + (alpha,))

        pygame.gfxdraw.aacircle(surface, x, y, radius, self.color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, self.color)

class Triangle(Shape):
    """A triangle created from a collision."""
    def __init__(self, x, y, color, area):
        super().__init__(x, y, color, area)
        self.side_length = math.sqrt((4 * self.area) / math.sqrt(3))
        self.type = "triangle"
        self.update_points()
        self.rect = pygame.Rect(self.pos.x - self.side_length/2, self.pos.y - self.side_length * math.sqrt(3)/3, self.side_length, self.side_length * math.sqrt(3)/2)

    def update_points(self, scale=1.0):
        s = self.side_length * scale
        h = s * math.sqrt(3) / 2
        p1 = (self.pos.x, self.pos.y - 2/3 * h)
        p2 = (self.pos.x - s/2, self.pos.y + 1/3 * h)
        p3 = (self.pos.x + s/2, self.pos.y + 1/3 * h)
        self.points = [p1, p2, p3]

    def draw(self, surface):
        scale = self.get_pulse_scale() * self.get_spawn_scale()
        if scale <= 0: return
        
        self.update_points(scale)
        int_points = [(int(p[0]), int(p[1])) for p in self.points]
        
        glow_color = tuple(min(255, c + 50) for c in self.color)
        
        # Glow effect
        for i in range(int(self.side_length * scale / 10), 0, -2):
             alpha = int(60 * (1 - (i / (self.side_length * scale / 10))))
             if alpha > 0:
                pygame.gfxdraw.aapolygon(surface, int_points, glow_color + (alpha,))

        pygame.gfxdraw.aapolygon(surface, int_points, self.color)
        pygame.gfxdraw.filled_polygon(surface, int_points, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Combine geometric shapes to create more complex forms and score points. "
        "Merge squares into circles, and circles into triangles before time runs out!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the red square. Hold 'space' to move the green square up, and 'shift' to move the blue square up. Collide shapes to score."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    WIN_SCORE = 1000
    PLAYER_SPEED = 6

    COLOR_BG = (15, 18, 28)
    COLOR_RED = (255, 87, 87)
    COLOR_GREEN = (87, 255, 150)
    COLOR_BLUE = (87, 150, 255)
    COLOR_UI = (240, 240, 255)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        self.render_mode = render_mode
        self.screen_bounds = self.screen.get_rect()

        # Initialize state variables
        self.player_agents = []
        self.entities = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        
        # self.reset() # This was causing issues with some test harnesses
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        self.player_agents = [
            PlayerSquare(self.WIDTH * 0.25, self.HEIGHT * 0.75, self.COLOR_RED, 1.0),
            PlayerSquare(self.WIDTH * 0.50, self.HEIGHT * 0.25, self.COLOR_GREEN, 1.5),
            PlayerSquare(self.WIDTH * 0.75, self.HEIGHT * 0.75, self.COLOR_BLUE, 2.0),
        ]
        self.entities = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        self.steps += 1
        self.timer -= 1
        
        # --- 1. Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game Entities ---
        for entity in self.player_agents + self.entities:
            entity.update()
        
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

        # --- 3. Collision Detection and Resolution ---
        reward = self._handle_collisions()

        # --- 4. Calculate Rewards & Check Termination ---
        terminated = self.timer <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Timeout penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Action 0 (Red Square)
        dx, dy = 0, 0
        if movement == 1: dy = -self.PLAYER_SPEED  # Up
        elif movement == 2: dy = self.PLAYER_SPEED   # Down
        elif movement == 3: dx = -self.PLAYER_SPEED  # Left
        elif movement == 4: dx = self.PLAYER_SPEED   # Right
        if dx != 0 or dy != 0:
            self.player_agents[0].move(dx, dy, self.screen_bounds)

        # Action 1 (Green Square)
        if space_held:
            # Find the green player agent to move
            for agent in self.player_agents:
                if agent.color == self.COLOR_GREEN:
                    agent.move(0, -self.PLAYER_SPEED, self.screen_bounds)
                    break

        # Action 2 (Blue Square)
        if shift_held:
            # Find the blue player agent to move
            for agent in self.player_agents:
                if agent.color == self.COLOR_BLUE:
                    agent.move(0, -self.PLAYER_SPEED, self.screen_bounds)
                    break

    def _handle_collisions(self):
        step_reward = 0
        all_shapes = self.player_agents + self.entities
        to_add = []
        to_remove = set()

        for s1, s2 in itertools.combinations(all_shapes, 2):
            if s1 in to_remove or s2 in to_remove:
                continue
            
            if s1.rect.colliderect(s2.rect):
                # --- Collision occurred ---
                
                to_remove.add(s1)
                to_remove.add(s2)

                new_area = s1.area + s2.area
                new_pos = (s1.pos * s1.area + s2.pos * s2.area) / new_area
                new_color = tuple(int((c1*s1.area + c2*s2.area)/new_area) for c1, c2 in zip(s1.color, s2.color))
                
                # Determine moving square for multiplier
                multiplier = 1.0
                if isinstance(s1, PlayerSquare): multiplier = s1.multiplier
                elif isinstance(s2, PlayerSquare): multiplier = s2.multiplier

                # Determine new shape type
                types = {s1.type, s2.type}
                new_shape = None
                if types == {"square"}:
                    new_shape = Circle(new_pos.x, new_pos.y, new_color, new_area)
                elif types == {"square", "circle"}:
                    new_shape = Triangle(new_pos.x, new_pos.y, new_color, new_area)
                elif types == {"circle"}:
                    new_shape = Circle(new_pos.x, new_pos.y, new_color, new_area)
                elif "triangle" in types:
                    new_shape = Triangle(new_pos.x, new_pos.y, new_color, new_area)
                
                if new_shape:
                    to_add.append(new_shape)
                    
                    # Calculate reward
                    area_reward = new_area / 1000 * multiplier  # Scaled area
                    transform_reward = 1.0
                    total_reward = area_reward + transform_reward
                    step_reward += total_reward
                    self.score += total_reward * 10 # Scale score for display

                    # Spawn particles
                    for _ in range(20):
                        self.particles.append(Particle(new_pos.x, new_pos.y, new_color))
        
        # Update entity lists
        if to_remove:
            self.player_agents = [p for p in self.player_agents if p not in to_remove]
            self.entities = [e for e in self.entities if e not in to_remove]
            self.entities.extend(to_add)

        return step_reward
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles first (background)
        for p in self.particles:
            p.draw(self.screen)

        # Render all shapes
        all_shapes = self.player_agents + self.entities
        for entity in sorted(all_shapes, key=lambda s: s.area):
             entity.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (20, 10))
        
        # Timer
        time_left_sec = self.timer / 30.0
        timer_color = self.COLOR_UI if time_left_sec > 10 else self.COLOR_RED
        timer_text = self.font_large.render(f"TIME: {time_left_sec:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME UP!"
            msg_render = self.font_large.render(msg, True, self.COLOR_UI)
            msg_rect = msg_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(msg_render, msg_rect)

            final_score_render = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI)
            final_score_rect = final_score_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_render, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "entities": len(self.player_agents) + len(self.entities)
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch back to a visible display driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.quit()
    pygame.init()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Shape Transformer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE:
                    space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        else:
            movement = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if term:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()