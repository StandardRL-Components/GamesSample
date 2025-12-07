
# Generated: 2025-08-28T06:29:29.147882
# Source Brief: brief_02941.md
# Brief Index: 2941

        
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

    user_guide = (
        "Use the arrow keys to aim your next track piece. Hold Space to draw a long piece, or Shift for a short one. "
        "Create a path for the rider to reach the goal!"
    )

    game_description = (
        "Draw a track for a physics-based sled to ride from the start to the finish line. "
        "Plan your route carefully to build up speed without crashing."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (220, 220, 220)
        self.COLOR_GRID = (200, 200, 200)
        self.COLOR_TRACK = (10, 10, 10)
        self.COLOR_SLED = (0, 120, 255)
        self.COLOR_START = (0, 200, 0)
        self.COLOR_FINISH = (255, 0, 0)
        self.COLOR_CURSOR = (255, 100, 0)
        self.COLOR_TEXT = (50, 50, 50)

        # Game constants
        self.MAX_STEPS = 500
        self.SIMULATION_SUBSTEPS = 50
        self.LINE_LENGTH_SHORT = 20
        self.LINE_LENGTH_MEDIUM = 40
        self.LINE_LENGTH_LONG = 80
        self.GRAVITY = pygame.math.Vector2(0, 0.04)
        self.FRICTION = 0.998
        self.CRASH_VELOCITY_LIMIT = 5.0
        self.CRASH_NORMAL_VEL_LIMIT = 1.5
        
        # State variables are initialized in reset()
        self.sled = None
        self.lines = None
        self.particles = None
        self.cursor_pos = None
        self.last_draw_vec = None
        self.start_pos = None
        self.finish_line_x = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.crashed = False
        
        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.crashed = False
        
        self.start_pos = pygame.math.Vector2(80, self.HEIGHT - 100)
        self.finish_line_x = self.WIDTH - 40

        self.sled = self._Sled(self.start_pos)
        self.particles = []
        
        # Initial track platform
        initial_line_start = pygame.math.Vector2(20, self.start_pos.y + 20)
        initial_line_end = pygame.math.Vector2(self.start_pos.x + 20, self.start_pos.y + 20)
        self.lines = [(initial_line_start, initial_line_end)]
        
        self.cursor_pos = pygame.math.Vector2(initial_line_end)
        self.last_draw_vec = pygame.math.Vector2(1, 0)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # 1. Unpack action and determine new line segment
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.last_draw_vec = pygame.math.Vector2(0, -1)
        elif movement == 2: self.last_draw_vec = pygame.math.Vector2(0, 1)
        elif movement == 3: self.last_draw_vec = pygame.math.Vector2(-1, 0)
        elif movement == 4: self.last_draw_vec = pygame.math.Vector2(1, 0)
        
        if movement == 0: # No-op: draw small horizontal line
            length = self.LINE_LENGTH_SHORT
            draw_vec = pygame.math.Vector2(1, 0)
        else:
            if space_held: length = self.LINE_LENGTH_LONG
            elif shift_held: length = self.LINE_LENGTH_SHORT
            else: length = self.LINE_LENGTH_MEDIUM
            draw_vec = self.last_draw_vec

        start_point = self.cursor_pos
        end_point = start_point + draw_vec * length
        
        # Clamp line to screen bounds to prevent drawing outside
        end_point.x = max(0, min(self.WIDTH, end_point.x))
        end_point.y = max(0, min(self.HEIGHT, end_point.y))
        
        self.lines.append((start_point, end_point))
        self.cursor_pos = end_point

        # 2. Run physics simulation
        self._run_simulation()
        
        # 3. Calculate reward and termination
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _run_simulation(self):
        for _ in range(self.SIMULATION_SUBSTEPS):
            if self.crashed or self.game_won:
                break
            self._update_sled_physics()
            self._check_sled_status()

    def _update_sled_physics(self):
        # Apply gravity
        self.sled.vel += self.GRAVITY
        
        # Move sled
        self.sled.pos += self.sled.vel

        # Collision detection and response
        contact_line, contact_point = self._find_contact_line()
        
        if contact_line:
            p1, p2 = contact_line
            line_vec = p2 - p1
            if line_vec.length() == 0: return

            # Crash detection
            if self.sled.vel.length() > self.CRASH_VELOCITY_LIMIT:
                self.crashed = True
                # SFX: Loud crash
                self._create_particles(self.sled.pos, self.COLOR_FINISH, 30)
                return

            line_normal = line_vec.rotate(90).normalize()
            if line_normal.y > 0: line_normal = -line_normal
            
            vel_into_normal = self.sled.vel.dot(line_normal)
            if vel_into_normal > self.CRASH_NORMAL_VEL_LIMIT:
                self.crashed = True
                # SFX: Bonk
                self._create_particles(self.sled.pos, self.COLOR_FINISH, 30)
                return
            
            # Response: project velocity away from normal and snap position
            if vel_into_normal > 0:
                self.sled.vel -= vel_into_normal * line_normal
            
            self.sled.pos.y = contact_point.y - 5 # Snap position slightly above line
            
            # Apply friction
            self.sled.vel *= self.FRICTION
            
            # Align sled to track
            target_angle = line_vec.angle_to(pygame.math.Vector2(1, 0))
            self.sled.angle = self._lerp_angle(self.sled.angle, target_angle, 0.2)
        else:
            # In freefall, slowly rotate to be horizontal
            self.sled.angle = self._lerp_angle(self.sled.angle, 0, 0.02)

    def _find_contact_line(self):
        best_y = float('inf')
        best_line = None
        contact_pt = None

        for line in self.lines:
            p1, p2 = line
            # Ensure p1 is left of p2 for consistent calculations
            if p1.x > p2.x: p1, p2 = p2, p1

            if p1.x <= self.sled.pos.x <= p2.x:
                if p2.x - p1.x == 0: # Vertical line
                    continue
                
                t = (self.sled.pos.x - p1.x) / (p2.x - p1.x)
                line_y = p1.y + t * (p2.y - p1.y)

                if self.sled.pos.y + 5 >= line_y and line_y < best_y:
                    best_y = line_y
                    best_line = line
                    contact_pt = pygame.math.Vector2(self.sled.pos.x, line_y)

        return best_line, contact_pt

    def _check_sled_status(self):
        # Win condition
        if self.sled.pos.x >= self.finish_line_x:
            self.game_won = True
            # SFX: Win jingle
            self._create_particles(self.sled.pos, self.COLOR_START, 50)
        
        # Crash condition (out of bounds)
        if not (0 < self.sled.pos.x < self.WIDTH and 0 < self.sled.pos.y < self.HEIGHT):
            self.crashed = True
            # SFX: Falling whistle then crash
            self._create_particles(self.sled.pos, self.COLOR_FINISH, 30)

    def _calculate_reward(self):
        if self.game_won: return 50.0
        if self.crashed: return -10.0
        return 0.1 # Survival reward

    def _check_termination(self):
        return self.game_won or self.crashed or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw start/finish lines
        start_y = self.start_pos.y + 20
        pygame.draw.line(self.screen, self.COLOR_START, (0, start_y), (self.start_pos.x + 20, start_y), 5)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.HEIGHT), 5)

        # Draw track lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

        # Draw particles
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)
            else:
                p.draw(self.screen)
        
        # Draw sled
        if not self.crashed:
            self.sled.draw(self.screen, self.COLOR_SLED)
        
        # Draw cursor
        if not self.game_over:
            pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 5, self.COLOR_CURSOR)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

        if self.game_over:
            msg = "GOAL!" if self.game_won else "CRASHED"
            color = self.COLOR_START if self.game_won else self.COLOR_FINISH
            end_text = self.font_msg.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "won": self.game_won,
            "crashed": self.crashed
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(self._Particle(pos, color, self.np_random))

    @staticmethod
    def _lerp_angle(a, b, t):
        diff = (b - a + 180) % 360 - 180
        return (a + diff * t) % 360

    def close(self):
        pygame.quit()

    class _Sled:
        def __init__(self, pos):
            self.pos = pygame.math.Vector2(pos)
            self.vel = pygame.math.Vector2(0, 0)
            self.angle = 0.0 # degrees
            self.size = 12

        def draw(self, surface, color):
            # Triangle shape for the sled
            p1 = pygame.math.Vector2(self.size, 0).rotate(-self.angle) + self.pos
            p2 = pygame.math.Vector2(-self.size/2, -self.size/2).rotate(-self.angle) + self.pos
            p3 = pygame.math.Vector2(-self.size/2, self.size/2).rotate(-self.angle) + self.pos
            points = [p1, p2, p3]
            
            pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in points], color)
            pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in points], color)

    class _Particle:
        def __init__(self, pos, color, np_random):
            self.pos = pygame.math.Vector2(pos)
            self.color = color
            angle = np_random.uniform(0, 2 * math.pi)
            speed = np_random.uniform(1, 5)
            self.vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.lifespan = np_random.integers(20, 40)
            self.size = np_random.integers(2, 5)

        def update(self):
            self.pos += self.vel
            self.vel *= 0.95 # Damping
            self.lifespan -= 1

        def is_dead(self):
            return self.lifespan <= 0

        def draw(self, surface):
            alpha = max(0, min(255, int(255 * (self.lifespan / 20))))
            color = self.color + (alpha,)
            pygame.draw.rect(surface, color, (int(self.pos.x), int(self.pos.y), self.size, self.size))

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        
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
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    game_over = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False

        if not game_over:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action
            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]
            
            # Since auto_advance is False, we only step when there's an input.
            # For a better human experience, we can step on any key press.
            if any(keys):
                obs, reward, terminated, truncated, info = env.step(action)
                game_over = terminated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit human play speed

    env.close()