import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:16:01.113627
# Source Brief: brief_02071.md
# Brief Index: 2071
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gravitas: A physics-based puzzle game Gymnasium environment.

    The player controls global gravity to guide colored shapes through
    matching colored gates to an exit portal. The goal is to get all
    shapes to the exit within a step limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the global gravity to guide shapes through matching colored gates and into the exit portal."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to change the direction of global gravity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1500
    GRAVITY_ACCEL = 0.1
    MAX_VELOCITY = 5.0
    FRICTION = 0.99
    SHAPE_RADIUS = 10
    EXIT_RADIUS = 20
    PARTICLE_LIFESPAN = 30

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_WALL = (80, 90, 110)
    COLOR_EXIT = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    SHAPE_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255),
        "white": (240, 240, 240)
    }
    GATE_COLORS = {
        "red": (200, 60, 60),
        "green": (60, 200, 60),
        "blue": (60, 60, 200),
        "white": (200, 200, 200)
    }
    
    LEVEL_CONFIGS = [
        # Level 0: Simple single shape, single gate
        {
            "walls": [((100, 250), (440, 20))],
            "gates": [{"pos": (300, 230), "size": (40, 20), "color_key": "red"}],
            "shapes": [{"pos": (320, 100), "color_key": "red"}],
            "exit_pos": (320, 320)
        },
        # Level 1: Two shapes, two gates
        {
            "walls": [((150, 0), (20, 300)), ((470, 100), (20, 300))],
            "gates": [
                {"pos": (150, 300), "size": (20, 40), "color_key": "red"},
                {"pos": (470, 60), "size": (20, 40), "color_key": "green"}
            ],
            "shapes": [
                {"pos": (100, 50), "color_key": "red"},
                {"pos": (540, 50), "color_key": "green"}
            ],
            "exit_pos": (320, 200)
        },
        # Level 2: Maze-like structure
        {
            "walls": [
                ((0, 150), (250, 20)), ((WIDTH - 250, 150), (250, 20)),
                ((150, 280), (340, 20))
            ],
            "gates": [
                {"pos": (250, 150), "size": (140, 20), "color_key": "blue"},
                {"pos": (130, 280), "size": (20, 40), "color_key": "red"}
            ],
            "shapes": [
                {"pos": (50, 50), "color_key": "red"},
                {"pos": (590, 50), "color_key": "blue"}
            ],
            "exit_pos": (320, 350)
        },
        # Level 3: Central block, requires careful navigation
        {
            "walls": [((220, 100), (200, 200))],
            "gates": [
                {"pos": (220, 80), "size": (200, 20), "color_key": "white"},
                {"pos": (200, 100), "size": (20, 200), "color_key": "green"}
            ],
            "shapes": [
                {"pos": (100, 200), "color_key": "green"},
                {"pos": (540, 200), "color_key": "red"}
            ],
            "exit_pos": (320, 350)
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.level = 0
        self.successful_resets = 0
        
        # self._initialize_state() is called in reset()
        
    def _initialize_state(self):
        """Initializes all mutable state variables for a new episode."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = np.array([0.0, 0.0])
        self.last_gravity_direction = 0
        self.shapes = []
        self.walls = []
        self.gates = []
        self.exit_pos = np.array([0.0, 0.0])
        self.particles = []
        self._generate_level()

    def _generate_level(self):
        """Creates the level layout, shapes, and gates."""
        config = self.LEVEL_CONFIGS[self.level % len(self.LEVEL_CONFIGS)]
        
        self.walls = [pygame.Rect(pos, size) for pos, size in config["walls"]]
        self.exit_pos = np.array(config["exit_pos"], dtype=float)
        
        self.gates = []
        for i, gate_cfg in enumerate(config["gates"]):
            self.gates.append({
                "id": i,
                "rect": pygame.Rect(gate_cfg["pos"], gate_cfg["size"]),
                "color_key": gate_cfg["color_key"],
                "color": self.GATE_COLORS[gate_cfg["color_key"]],
                "is_open": False,
                "open_animation": 0.0
            })
            
        self.shapes = []
        num_shapes = min(5, 1 + self.successful_resets // 5)
        available_shapes = config["shapes"] * ( (num_shapes // len(config["shapes"])) + 1)

        for i in range(num_shapes):
            shape_cfg = available_shapes[i]
            pos = np.array(shape_cfg["pos"], dtype=float) + self.np_random.uniform(-5, 5, size=2)
            dist_to_exit = np.linalg.norm(pos - self.exit_pos)
            self.shapes.append({
                "id": i,
                "pos": pos,
                "vel": np.array([0.0, 0.0]),
                "color_key": shape_cfg["color_key"],
                "color": self.SHAPE_COLORS[shape_cfg["color_key"]],
                "last_dist_to_exit": dist_to_exit
            })
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, _, _ = action
        reward = 0
        self.steps += 1

        # 1. Update Gravity
        if movement != self.last_gravity_direction:
            if movement == 0: self.gravity = np.array([0.0, 0.0])
            elif movement == 1: self.gravity = np.array([0, -self.GRAVITY_ACCEL])
            elif movement == 2: self.gravity = np.array([0, self.GRAVITY_ACCEL])
            elif movement == 3: self.gravity = np.array([-self.GRAVITY_ACCEL, 0])
            elif movement == 4: self.gravity = np.array([self.GRAVITY_ACCEL, 0])
            self.last_gravity_direction = movement
        
        # 2. Update Shapes
        exited_shapes = []
        for shape in self.shapes:
            # Physics
            shape["vel"] += self.gravity
            shape["vel"] *= self.FRICTION
            vel_mag = np.linalg.norm(shape["vel"])
            if vel_mag > self.MAX_VELOCITY:
                shape["vel"] = shape["vel"] / vel_mag * self.MAX_VELOCITY
            shape["pos"] += shape["vel"]
            
            # Screen wrap
            shape["pos"][0] %= self.WIDTH
            shape["pos"][1] %= self.HEIGHT
            
            # Collisions
            self._handle_collisions(shape)
            
            # Gate interaction
            reward += self._handle_gate_interaction(shape)

            # Distance reward
            dist = np.linalg.norm(shape["pos"] - self.exit_pos)
            if dist < shape["last_dist_to_exit"]:
                reward += 0.01
            else:
                reward -= 0.001
            shape["last_dist_to_exit"] = dist

            # Check for exit
            if dist < self.EXIT_RADIUS:
                exited_shapes.append(shape)
                self._create_particles(shape["pos"], shape["color"], 50, 2.0)
                # SFX: Shape exit sound

        # Remove exited shapes
        if exited_shapes:
            self.shapes = [s for s in self.shapes if s not in exited_shapes]

        # 3. Update Particles & Gates
        self._update_particles()
        for gate in self.gates:
            if gate["is_open"] and gate["open_animation"] < 1.0:
                gate["open_animation"] = min(1.0, gate["open_animation"] + 0.1)

        # 4. Check Termination
        terminated = False
        truncated = False
        win = not self.shapes
        timeout = self.steps >= self.MAX_STEPS

        if win:
            reward += 100
            terminated = True
            self.game_over = True
            self.successful_resets += 1
            if self.successful_resets % 2 == 0:
                 self.level = (self.level + 1) % len(self.LEVEL_CONFIGS)
        elif timeout:
            reward -= 10
            terminated = True # Per Gymnasium API, timeout is a termination, not truncation
            self.game_over = True
            self.successful_resets = 0

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_collisions(self, shape):
        shape_rect = pygame.Rect(shape["pos"] - self.SHAPE_RADIUS, (self.SHAPE_RADIUS * 2, self.SHAPE_RADIUS * 2))
        
        collidable_rects = self.walls + [g["rect"] for g in self.gates if not g["is_open"]]

        for rect in collidable_rects:
            if shape_rect.colliderect(rect):
                # Simple push-out collision response
                overlap_x = (shape_rect.width / 2 + rect.width / 2) - abs(shape_rect.centerx - rect.centerx)
                overlap_y = (shape_rect.height / 2 + rect.height / 2) - abs(shape_rect.centery - rect.centery)

                if overlap_x > 0 and overlap_y > 0:
                    self._create_particles(shape["pos"], (200, 200, 200), 5, 1.0) # Wall collision sparks
                    # SFX: Thud sound
                    if overlap_x < overlap_y:
                        if shape_rect.centerx < rect.centerx:
                            shape["pos"][0] -= overlap_x
                        else:
                            shape["pos"][0] += overlap_x
                        shape["vel"][0] *= -0.8
                    else:
                        if shape_rect.centery < rect.centery:
                            shape["pos"][1] -= overlap_y
                        else:
                            shape["pos"][1] += overlap_y
                        shape["vel"][1] *= -0.8

    def _handle_gate_interaction(self, shape):
        reward = 0
        shape_rect = pygame.Rect(shape["pos"] - self.SHAPE_RADIUS, (self.SHAPE_RADIUS * 2, self.SHAPE_RADIUS * 2))
        for gate in self.gates:
            if not gate["is_open"] and shape_rect.colliderect(gate["rect"]):
                if gate["color_key"] == "white" or gate["color_key"] == shape["color_key"]:
                    gate["is_open"] = True
                    reward += 5.0
                    self._create_particles(gate["rect"].center, gate["color"], 30, 1.5)
                    # SFX: Gate open chime
        return reward

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "shapes_remaining": len(self.shapes),
            "consecutive_wins": self.successful_resets
        }
        
    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": vel,
                "color": color,
                "lifespan": self.PARTICLE_LIFESPAN,
                "max_lifespan": self.PARTICLE_LIFESPAN
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_exit()
        self._render_walls()
        self._render_gates()
        self._render_particles()
        self._render_shapes()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_exit(self):
        pulse = (math.sin(self.steps * 0.05) + 1) / 2
        pos_int = (int(self.exit_pos[0]), int(self.exit_pos[1]))
        
        for i in range(4):
            radius = self.EXIT_RADIUS * (1 + pulse * 0.2) - i * 4
            alpha = 100 - i * 20
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), (*self.COLOR_EXIT, alpha))
        
    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_gates(self):
        for gate in self.gates:
            if gate["is_open"]:
                # Draw as two separate posts
                anim_progress = gate["open_animation"]
                if gate["rect"].width > gate["rect"].height: # Horizontal gate
                    w = gate["rect"].width / 2 * (1 - anim_progress)
                    post1 = pygame.Rect(gate["rect"].left, gate["rect"].top, w, gate["rect"].height)
                    post2 = pygame.Rect(gate["rect"].right - w, gate["rect"].top, w, gate["rect"].height)
                    pygame.draw.rect(self.screen, gate["color"], post1, border_radius=2)
                    pygame.draw.rect(self.screen, gate["color"], post2, border_radius=2)
                else: # Vertical gate
                    h = gate["rect"].height / 2 * (1 - anim_progress)
                    post1 = pygame.Rect(gate["rect"].left, gate["rect"].top, gate["rect"].width, h)
                    post2 = pygame.Rect(gate["rect"].left, gate["rect"].bottom - h, gate["rect"].width, h)
                    pygame.draw.rect(self.screen, gate["color"], post1, border_radius=2)
                    pygame.draw.rect(self.screen, gate["color"], post2, border_radius=2)
            else:
                # Draw as a solid block with a lock icon
                pygame.draw.rect(self.screen, gate["color"], gate["rect"], border_radius=2)
                c = gate["rect"].center
                pygame.gfxdraw.filled_circle(self.screen, c[0], c[1] + 2, 4, (0,0,0,50))
                pygame.gfxdraw.aacircle(self.screen, c[0], c[1] + 2, 4, (255,255,255,50))


    def _render_shapes(self):
        for shape in self.shapes:
            pos_int = (int(shape["pos"][0]), int(shape["pos"][1]))
            # Glow effect
            glow_color = (*shape["color"], 60)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.SHAPE_RADIUS + 2, glow_color)
            # Main shape
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.SHAPE_RADIUS, shape["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.SHAPE_RADIUS, shape["color"])

    def _render_particles(self):
        for p in self.particles:
            pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            radius = int(self.SHAPE_RADIUS / 3 * (p["lifespan"] / p["max_lifespan"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        shapes_text = self.font_small.render(f"SHAPES: {len(self.shapes)}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 25))
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))
        self.screen.blit(shapes_text, (self.WIDTH - shapes_text.get_width() - 10, 25))
        
        # Gravity indicator
        gravity_arrows = {0: " ", 1: "↑", 2: "↓", 3: "←", 4: "→"}
        grav_text = self.font_small.render(f"GRAVITY: {gravity_arrows[self.last_gravity_direction]}", True, self.COLOR_TEXT)
        self.screen.blit(grav_text, (self.WIDTH/2 - grav_text.get_width()/2, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        message = "SUCCESS!" if not self.shapes else "TIME UP"
        color = self.SHAPE_COLORS["green"] if not self.shapes else self.SHAPE_COLORS["red"]
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To run manual play, you need a display.
    # Comment out or remove `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    # at the top of the file.
    try:
        pygame.display.set_caption("Gravitas - Manual Control")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    except pygame.error as e:
        print(f"Could not set up display for manual play: {e}")
        print("This is expected if you are running in a headless environment.")
        print("To run with a display, comment out the `SDL_VIDEODRIVER` line at the top of the file.")
        env.close()
        exit()

    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print("Controls: Arrow keys to change gravity. R to reset.")
    
    while True:
        action = np.array([0, 0, 0]) # Default action: no gravity change
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Pygame uses a different coordinate system for display
        # Transpose from (H, W, C) to (W, H, C) for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS