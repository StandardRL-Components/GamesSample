import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:07:02.946521
# Source Brief: brief_01528.md
# Brief Index: 1528
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Functions and Classes ---

def get_line_segment_intersection(p1, p2, p3, p4):
    """
    Calculates the intersection point of two line segments.
    Returns the intersection point as a Vector2, or None if they don't intersect.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 < t < 1 and 0 < u < 1:
        # Using 0 < t < 1 to avoid issues with vertices
        pt_x = x1 + t * (x2 - x1)
        pt_y = y1 + t * (y2 - y1)
        return pygame.Vector2(pt_x, pt_y)

    return None

def point_line_distance(point, line_start, line_end):
    """Calculates the perpendicular distance from a point to a line segment."""
    p = pygame.Vector2(point)
    a = pygame.Vector2(line_start)
    b = pygame.Vector2(line_end)
    
    if a == b:
        return (p - a).length()
    
    ab = b - a
    ap = p - a
    
    proj = ap.dot(ab) / ab.length_squared()
    
    if proj < 0:
        return ap.length()
    elif proj > 1:
        return (p - b).length()
    else:
        closest_point = a + proj * ab
        return (p - closest_point).length()


class Particle:
    def __init__(self, pos, vel, color, size, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98  # Damping
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifespan))
            # Draw a simple circle for the particle
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y), 
                int(self.size), (*self.color, alpha)
            )


class Shape:
    MIN_SIZE = 15
    FADE_DURATION = 15

    def __init__(self, shape_type, pos, size, color, fall_speed):
        self.shape_type = shape_type
        self.pos = pygame.Vector2(pos)
        self.size = size
        self.color = color
        self.vel = pygame.Vector2(0, fall_speed)
        self.angle = random.uniform(0, 2 * math.pi)
        self.rot_speed = random.uniform(-0.02, 0.02)
        self.vertices = self._generate_vertices()
        self.fading = False
        self.fade_timer = self.FADE_DURATION

    def _generate_vertices(self):
        verts = []
        if self.shape_type == "square":
            half = self.size / 2
            verts = [(-half, -half), (half, -half), (half, half), (-half, half)]
        elif self.shape_type == "triangle":
            height = self.size * math.sqrt(3) / 2
            verts = [(0, -height / 2), (-self.size / 2, height / 2), (self.size / 2, height / 2)]
        
        return [pygame.Vector2(v) for v in verts]

    def get_world_vertices(self):
        """ Returns vertices in world coordinates (rotated and translated) """
        rotated_verts = [v.rotate_rad(self.angle) for v in self.vertices]
        return [v + self.pos for v in rotated_verts]

    def update(self, fall_speed):
        if not self.fading:
            self.vel.y = fall_speed
            self.pos += self.vel
            self.angle += self.rot_speed
        else:
            self.fade_timer -= 1
        return self.fade_timer > 0

    def draw(self, surface):
        alpha = 255
        if self.fading:
            alpha = int(255 * max(0, self.fade_timer / self.FADE_DURATION))

        glow_color = (*self.color, int(alpha / 8))

        if self.shape_type == "circle":
            for i in range(4, 0, -1):
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size + i * 2), glow_color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.size), (*self.color, alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), (*self.color, alpha))
        else: # Polygons
            world_verts = self.get_world_vertices()
            if len(world_verts) < 3: return
            
            int_verts = [(int(v.x), int(v.y)) for v in world_verts]
            
            for i in range(4, 0, -1):
                 pygame.draw.polygon(surface, glow_color, [(v.x, v.y) for v in self.get_world_vertices()], width=i*4)

            pygame.gfxdraw.aapolygon(surface, int_verts, (*self.color, alpha))
            pygame.gfxdraw.filled_polygon(surface, int_verts, (*self.color, alpha))

    def start_fading(self):
        self.fading = True


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Slice falling geometric shapes with a rotating laser. Split shapes into smaller pieces to "
        "score points, but don't let them fall off the screen!"
    )
    user_guide = "Controls: Use ← and → arrow keys to rotate the laser. Press space to fire and slice the shapes."
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    FPS = 30
    MAX_EPISODE_STEPS = 2000
    WIN_SCORE = 100
    INITIAL_SCORE = 20
    
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_LASER = (255, 255, 255)
    COLOR_UI = (220, 220, 255)
    SHAPE_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)] # Cyan, Magenta, Yellow

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)

        # Game state variables are initialized in reset()
        self.laser_angle = 0.0
        self.laser_fire_timer = 0
        self.shapes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fall_speed = 0.0
        self.prev_space_held = False
        self.game_over = False
        
        self.reset()
        # self.validate_implementation() # Optional validation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = self.INITIAL_SCORE
        self.game_over = False
        
        self.laser_angle = -math.pi / 2  # Pointing up
        self.laser_fire_timer = 0
        self.prev_space_held = False
        
        self.shapes = []
        self.particles = []
        self.fall_speed = 1.0

        for _ in range(5):
            self._spawn_shape()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0
        
        # --- Handle Actions ---
        self._handle_input(movement, space_held)

        # --- Fire Laser ---
        fired_this_step = space_held and not self.prev_space_held
        if fired_this_step:
            # sfx: laser_fire.wav
            self.laser_fire_timer = 5 # Frames
            slice_reward = self._perform_slice()
            reward += slice_reward
        self.prev_space_held = space_held
        
        # --- Update Game State ---
        self._update_shapes()
        self._update_particles()
        
        # --- Spawn new shapes ---
        if self.steps % 45 == 0:
            self._spawn_shape()

        # --- Update difficulty ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed += 0.05
            
        # --- Decrement timers and increment steps ---
        self.laser_fire_timer = max(0, self.laser_fire_timer - 1)
        self.steps += 1
        
        # --- Calculate Reward and Termination ---
        reward += self._calculate_reward()
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.score <= 0:
                reward -= 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        rotation_speed = math.radians(2.0) # More responsive than 1 degree
        if movement == 3: # Left
            self.laser_angle -= rotation_speed
        elif movement == 4: # Right
            self.laser_angle += rotation_speed
        self.laser_angle %= (2 * math.pi)

    def _perform_slice(self):
        slice_reward = 0
        laser_dir = pygame.Vector2(math.cos(self.laser_angle), math.sin(self.laser_angle))
        p1 = self.CENTER - laser_dir * 1000
        p2 = self.CENTER + laser_dir * 1000

        new_shapes = []
        shapes_to_remove = []

        for shape in self.shapes:
            if shape.fading: continue

            if shape.shape_type == "circle":
                dist = point_line_distance(shape.pos, p1, p2)
                if dist < shape.size and shape.size > Shape.MIN_SIZE:
                    # sfx: slice_circle.wav
                    shape.start_fading()
                    shapes_to_remove.append(shape)
                    slice_reward += 1.0 # Simple reward for circles
                    
                    # Spawn two smaller circles
                    offset = laser_dir.rotate(90) * (shape.size * 0.5)
                    for sign in [-1, 1]:
                        new_pos = shape.pos + offset * sign
                        new_size = shape.size * 0.6
                        if new_size > Shape.MIN_SIZE / 2:
                            new_shape = Shape("circle", new_pos, new_size, shape.color, self.fall_speed)
                            new_shape.vel = shape.vel + offset.normalize() * sign * 2
                            new_shapes.append(new_shape)
                    
                    self._create_particle_burst(shape.pos, shape.color)
            
            else: # Polygons
                world_verts = shape.get_world_vertices()
                intersections = []
                intersected_edges = []

                for i in range(len(world_verts)):
                    v1 = world_verts[i]
                    v2 = world_verts[(i + 1) % len(world_verts)]
                    intersect_pt = get_line_segment_intersection(p1, p2, v1, v2)
                    if intersect_pt:
                        intersections.append(intersect_pt)
                        intersected_edges.append(i)

                if len(intersections) == 2 and shape.size > Shape.MIN_SIZE:
                    # sfx: slice_poly.wav
                    shape.start_fading()
                    shapes_to_remove.append(shape)

                    # Reward for perfect bisection
                    dist_to_center = point_line_distance(shape.pos, p1, p2)
                    if dist_to_center < shape.size * 0.1:
                        slice_reward += 20.0 # Perfect slice
                    else:
                        slice_reward += 1.0 # Partial slice

                    self._create_particle_burst((intersections[0] + intersections[1]) / 2, shape.color)

                    # Polygon splitting logic
                    v_list = list(range(len(world_verts)))
                    i1, i2 = intersected_edges
                    
                    # Sort edges to handle vertex ordering
                    if i1 > i2: i1, i2 = i2, i1
                    
                    p1_verts = [intersections[0]]
                    curr = (i1 + 1) % len(v_list)
                    while curr != (i2 + 1) % len(v_list):
                        p1_verts.append(world_verts[curr])
                        curr = (curr + 1) % len(v_list)
                    p1_verts.append(intersections[1])

                    p2_verts = [intersections[1]]
                    curr = (i2 + 1) % len(v_list)
                    while curr != (i1 + 1) % len(v_list):
                        p2_verts.append(world_verts[curr])
                        curr = (curr + 1) % len(v_list)
                    p2_verts.append(intersections[0])

                    for poly_verts in [p1_verts, p2_verts]:
                        if len(poly_verts) > 2:
                            new_pos = sum(poly_verts, pygame.Vector2()) / len(poly_verts)
                            max_dist = max((v - new_pos).length() for v in poly_verts)
                            new_shape = Shape("polygon", new_pos, max_dist * 1.5, shape.color, self.fall_speed)
                            new_shape.vertices = [v - new_pos for v in poly_verts]
                            new_shape.vel = shape.vel + (new_pos - shape.pos).normalize() * 1.5
                            new_shapes.append(new_shape)

        # Update shape list
        self.shapes = [s for s in self.shapes if s not in shapes_to_remove]
        self.shapes.extend(new_shapes)
        
        self.score += slice_reward
        return slice_reward

    def _update_shapes(self):
        surviving_shapes = []
        for shape in self.shapes:
            if shape.update(self.fall_speed):
                # Check for out of bounds (bottom of screen)
                if shape.pos.y - shape.size > self.SCREEN_HEIGHT:
                    # sfx: shape_miss.wav
                    self.score -= max(1, int(shape.size / 10))
                else:
                    surviving_shapes.append(shape)
        self.shapes = surviving_shapes

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _spawn_shape(self):
        shape_type = random.choice(["square", "triangle", "circle"])
        pos = pygame.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50), -50)
        size = random.uniform(25, 50)
        color = random.choice(self.SHAPE_COLORS)
        self.shapes.append(Shape(shape_type, pos, size, color, self.fall_speed))

    def _create_particle_burst(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = random.uniform(1, 4)
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, color, size, lifespan))

    def _calculate_reward(self):
        # Continuous reward for staying in the game
        return 0.1 if self.score > 0 else 0.0

    def _check_termination(self):
        return (
            self.score <= 0 or
            self.score >= self.WIN_SCORE or
            self.steps >= self.MAX_EPISODE_STEPS
        )
    
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
            "fall_speed": self.fall_speed,
        }

    def _render_game(self):
        # Render laser origin
        for i in range(8, 0, -1):
            pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 5 + i, (*self.COLOR_LASER, 15))
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 5, self.COLOR_LASER)

        # Render aiming reticle
        laser_dir = pygame.Vector2(math.cos(self.laser_angle), math.sin(self.laser_angle))
        end_point = self.CENTER + laser_dir * 25
        pygame.draw.line(self.screen, (*self.COLOR_LASER, 100), self.CENTER, end_point, 1)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Render shapes
        for s in self.shapes:
            s.draw(self.screen)

        # Render laser fire
        if self.laser_fire_timer > 0:
            alpha = int(255 * (self.laser_fire_timer / 5.0))
            color = (*self.COLOR_LASER, alpha)
            p1 = self.CENTER - laser_dir * 1000
            p2 = self.CENTER + laser_dir * 1000
            
            # Glow effect for laser
            pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha/4)), p1, p2, 7)
            pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha/2)), p1, p2, 3)
            pygame.draw.line(self.screen, color, p1, p2, 1)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        if self.game_over:
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 128) if self.score >= self.WIN_SCORE else (255, 64, 64)
            try:
                big_font = pygame.font.SysFont("monospace", 72, bold=True)
            except pygame.error:
                big_font = pygame.font.Font(None, 80)
            
            end_surf = big_font.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    # This block will not run in the test environment, but is useful for local development.
    # It requires a display to be available.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Laser Slicer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # None
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        if keys[pygame.K_r]:
            obs, info = env.reset()

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Game over, wait for reset
            pass

        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()