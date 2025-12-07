import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:31:47.371768
# Source Brief: brief_01623.md
# Brief Index: 1623
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent connects matching colored nodes.
    The agent controls a line drawer with momentum, spending a 'momentum'
    resource to draw lines. Connecting a node to itself creates a recursive
    sub-puzzle. The goal is to connect 5 pairs of nodes before running out
    of momentum.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Connect matching colored nodes by drawing a line with a cursor that has momentum. "
        "Connecting a node to itself creates a recursive sub-puzzle. Solve all connections before momentum runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a node and space again to connect. "
        "Press shift to cancel an active line."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    NUM_PAIRS = 5
    MAX_RECURSION = 3

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 58)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_MOMENTUM_HIGH = (60, 255, 120)
    COLOR_MOMENTUM_LOW = (255, 60, 60)
    NODE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_connections = 0
        self.puzzle_stack = []
        self.particles = []

        # Button state for edge detection
        self.prev_space_held = False
        self.prev_shift_held = False

        # Placeholder for active puzzle state attributes
        self._active_puzzle = None
        self.line_drawer = None
        
        # Initialize state variables
        # self.reset() # reset is called by the environment runner
        
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_connections = 0
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        # Create the initial puzzle
        self.puzzle_stack = [self._create_puzzle_state(level=0)]
        self._set_active_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Game Logic ---
        reward += self._handle_input(movement, space_pressed, shift_pressed)
        self._update_game_state()
        self._update_particles()
        
        # --- Reward Calculation ---
        reward += self._calculate_continuous_reward()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.total_connections >= self.NUM_PAIRS:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods: Game Logic ---

    def _set_active_puzzle(self):
        """Sets the currently active puzzle from the top of the stack."""
        if not self.puzzle_stack:
            self.game_over = True
            return

        self._active_puzzle = self.puzzle_stack[-1]
        self.line_drawer = {
            "active": False,
            "start_node": None,
            "path": [],
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float),
            "vel": np.array([0.0, 0.0], dtype=float),
        }

    def _create_puzzle_state(self, level=0, bounds=None):
        """Generates a new set of nodes for a puzzle."""
        if bounds is None:
            bounds = (40, 40, self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 80)
        
        bx, by, bw, bh = bounds
        nodes = []
        node_radius = max(8, 18 - level * 3)
        min_dist = (node_radius * 2 + 10) ** 2 # Use squared distance for efficiency

        for i in range(self.NUM_PAIRS):
            for _ in range(2): # Create a pair
                while True:
                    pos = np.array([
                        self.np_random.uniform(bx + node_radius, bx + bw - node_radius),
                        self.np_random.uniform(by + node_radius, by + bh - node_radius)
                    ])
                    # Ensure no overlap with existing nodes
                    if all(np.sum((pos - n['pos'])**2) > min_dist for n in nodes):
                        nodes.append({
                            "pos": pos,
                            "color": self.NODE_COLORS[i],
                            "pair_id": i,
                            "is_connected": False,
                            "radius": node_radius,
                            "id": len(nodes)
                        })
                        break
        
        return {
            "nodes": nodes,
            "connections": [],
            "momentum": 100.0,
            "bounds": bounds,
            "level": level
        }

    def _handle_input(self, movement, space_pressed, shift_pressed):
        """Process player actions and return event-based rewards."""
        reward = 0
        
        # --- Cancel Action ---
        if shift_pressed and self.line_drawer["active"]:
            # SFX: Cancel sound
            self.line_drawer["active"] = False
            self.line_drawer["start_node"] = None
            self.line_drawer["path"] = []
            reward -= 0.5 # Small penalty for canceling

        # --- Select/Connect Action ---
        if space_pressed:
            highlighted_node = self._get_highlighted_node()
            if highlighted_node:
                if not self.line_drawer["active"]:
                    # Select a start node
                    if not highlighted_node["is_connected"]:
                        # SFX: Select node sound
                        self.line_drawer["active"] = True
                        self.line_drawer["start_node"] = highlighted_node
                        self.line_drawer["path"] = [highlighted_node["pos"].copy()]
                        self.line_drawer["pos"] = highlighted_node["pos"].copy()
                        self.line_drawer["vel"] *= 0 # Reset velocity on select
                else:
                    # Attempt to connect to an end node
                    start_node = self.line_drawer["start_node"]
                    end_node = highlighted_node
                    
                    # Consume momentum based on path length
                    path_len = sum(np.linalg.norm(self.line_drawer["path"][i] - self.line_drawer["path"][i-1]) for i in range(1, len(self.line_drawer["path"])))
                    self._active_puzzle["momentum"] -= path_len / 10.0
                    
                    # Case 1: Connect to self (create sub-puzzle)
                    if start_node["id"] == end_node["id"]:
                        if self._active_puzzle["level"] < self.MAX_RECURSION:
                            # SFX: Sub-puzzle creation sound
                            self._create_sub_puzzle(start_node)
                        else: # Max recursion reached, treat as failed connection
                            reward -= 1
                    
                    # Case 2: Connect to another node
                    elif not end_node["is_connected"]:
                        # Correct Pair
                        if start_node["pair_id"] == end_node["pair_id"]:
                            # SFX: Success sound
                            reward += 10
                            self.total_connections += 1
                            start_node["is_connected"] = True
                            end_node["is_connected"] = True
                            self._active_puzzle["connections"].append((start_node, end_node))
                            self._spawn_particles(end_node["pos"], end_node["color"])
                        # Mismatched Pair
                        else:
                            # SFX: Failure sound
                            reward -= 1
                    
                    # Reset drawer after any connection attempt
                    self.line_drawer["active"] = False
                    self.line_drawer["start_node"] = None
                    self.line_drawer["path"] = []

        # --- Movement Control ---
        if self.line_drawer["active"]:
            accel = np.array([0.0, 0.0])
            move_force = 0.6
            if movement == 1: accel[1] -= move_force # Up
            if movement == 2: accel[1] += move_force # Down
            if movement == 3: accel[0] -= move_force # Left
            if movement == 4: accel[0] += move_force # Right
            self.line_drawer["vel"] += accel
        
        return reward

    def _update_game_state(self):
        """Updates positions and other state variables each step."""
        # Update line drawer with momentum physics
        drawer = self.line_drawer
        if drawer["active"]:
            drawer["vel"] *= 0.92  # Drag
            drawer["pos"] += drawer["vel"]

            # Clamp to screen bounds
            bounds = self._active_puzzle["bounds"]
            drawer["pos"][0] = np.clip(drawer["pos"][0], bounds[0], bounds[0] + bounds[2])
            drawer["pos"][1] = np.clip(drawer["pos"][1], bounds[1], bounds[1] + bounds[3])
            
            # Add new point to path if it has moved enough
            if not drawer["path"] or np.linalg.norm(drawer["pos"] - drawer["path"][-1]) > 2:
                drawer["path"].append(drawer["pos"].copy())
        
        # Check for puzzle completion
        if all(n["is_connected"] for n in self._active_puzzle["nodes"]):
            if self._active_puzzle["level"] > 0: # Sub-puzzle complete
                self.puzzle_stack.pop()
                self._set_active_puzzle()
                # Mark the node that spawned the sub-puzzle as connected
                # This is a simplification from the brief for cleaner game flow
                # It now counts as one of the 5 required connections
                self.total_connections += 1 
                self.score += 5 # Reward for completing sub-puzzle
            # else: main puzzle is complete, handled by termination check

    def _create_sub_puzzle(self, node):
        """Generates and activates a new sub-puzzle."""
        level = self._active_puzzle["level"] + 1
        size = 200 - level * 20
        bounds = (
            node["pos"][0] - size / 2,
            node["pos"][1] - size / 2,
            size,
            size
        )
        # Clamp bounds to be within screen
        bounds = (
            np.clip(bounds[0], 0, self.SCREEN_WIDTH - size),
            np.clip(bounds[1], 0, self.SCREEN_HEIGHT - size),
            size, size
        )

        sub_puzzle = self._create_puzzle_state(level, bounds)
        self.puzzle_stack.append(sub_puzzle)
        self._set_active_puzzle()

    def _get_highlighted_node(self):
        """Finds the node closest to the line drawer's head."""
        cursor_pos = self.line_drawer["pos"]
        min_dist_sq = float('inf')
        highlighted_node = None
        
        for node in self._active_puzzle["nodes"]:
            if node.get("is_connected"): continue
            dist_sq = np.sum((node["pos"] - cursor_pos)**2)
            # Highlight radius is larger than node radius
            if dist_sq < (node["radius"] * 2.5) ** 2 and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                highlighted_node = node
        return highlighted_node

    def _calculate_continuous_reward(self):
        """Provides a small reward for moving the line drawer towards the correct target."""
        if not self.line_drawer["active"]:
            return 0
        
        start_node = self.line_drawer["start_node"]
        target_node = None
        for node in self._active_puzzle["nodes"]:
            if node["pair_id"] == start_node["pair_id"] and not node["is_connected"] and node["id"] != start_node["id"]:
                target_node = node
                break
        
        if target_node:
            pos = self.line_drawer["pos"]
            prev_pos = self.line_drawer["path"][-2] if len(self.line_drawer["path"]) > 1 else pos
            
            dist_current = np.linalg.norm(pos - target_node["pos"])
            dist_prev = np.linalg.norm(prev_pos - target_node["pos"])
            
            # Reward is proportional to how much closer we got
            return (dist_prev - dist_current) * 0.01
        
        return 0

    def _check_termination(self):
        """Checks for win, loss, or max_steps termination."""
        if self.total_connections >= self.NUM_PAIRS:
            return True # Win
        if self._active_puzzle["momentum"] <= 0 and self._active_puzzle["level"] == 0:
            return True # Loss
        return False

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_puzzles()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
    
    def _render_puzzles(self):
        # Render parent puzzles semi-transparently
        for i, puzzle in enumerate(self.puzzle_stack):
            is_active = (i == len(self.puzzle_stack) - 1)
            alpha = 255 if is_active else 50
            
            # Render puzzle bounds for sub-puzzles
            if puzzle["level"] > 0:
                bounds_rect = pygame.Rect(*puzzle["bounds"])
                s = pygame.Surface((bounds_rect.w, bounds_rect.h), pygame.SRCALPHA)
                s.fill((40, 45, 70, alpha // 2))
                pygame.draw.rect(s, (*self.COLOR_GRID, alpha), s.get_rect(), 2)
                self.screen.blit(s, bounds_rect.topleft)

            # Render connections
            for start_node, end_node in puzzle["connections"]:
                color = start_node["color"]
                pygame.draw.aaline(self.screen, color, start_node["pos"], end_node["pos"], 2)

            # Render nodes
            for node in puzzle["nodes"]:
                self._draw_glowing_circle(
                    self.screen, node["pos"], node["radius"], node["color"], 
                    is_connected=node["is_connected"], alpha=alpha
                )
        
        # Render active line drawer on top
        if self.line_drawer["active"] and len(self.line_drawer["path"]) > 1:
            pygame.draw.aalines(self.screen, (255, 255, 255), False, self.line_drawer["path"], 2)
        
        # Render highlight on top of everything
        highlighted_node = self._get_highlighted_node()
        if highlighted_node:
            self._draw_glowing_circle(
                self.screen, highlighted_node["pos"], highlighted_node["radius"], 
                highlighted_node["color"], is_highlighted=True
            )

    def _draw_glowing_circle(self, surface, pos, radius, color, is_connected=False, is_highlighted=False, alpha=255):
        pos_int = (int(pos[0]), int(pos[1]))
        
        if is_highlighted:
            # Pulsing glow effect for highlighted node
            glow_radius = radius + 8 + math.sin(self.steps * 0.2) * 2
            glow_color = (*color, 60)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(glow_radius), glow_color)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(glow_radius), glow_color)
        
        if is_connected:
            main_color = (int(c*0.5) for c in color) # Darken connected nodes
            main_color_alpha = (*main_color, alpha)
        else:
            main_color_alpha = (*color, alpha)

        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), main_color_alpha)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), main_color_alpha)

    def _render_ui(self):
        # --- Momentum Bar ---
        momentum = self._active_puzzle["momentum"]
        bar_width = 200
        bar_height = 15
        fill_width = int((momentum / 100.0) * bar_width)
        
        # Interpolate color
        r = self.COLOR_MOMENTUM_LOW[0] + (self.COLOR_MOMENTUM_HIGH[0] - self.COLOR_MOMENTUM_LOW[0]) * (momentum / 100.0)
        g = self.COLOR_MOMENTUM_LOW[1] + (self.COLOR_MOMENTUM_HIGH[1] - self.COLOR_MOMENTUM_LOW[1]) * (momentum / 100.0)
        b = self.COLOR_MOMENTUM_LOW[2] + (self.COLOR_MOMENTUM_HIGH[2] - self.COLOR_MOMENTUM_LOW[2]) * (momentum / 100.0)
        bar_color = (int(r), int(g), int(b))

        pygame.draw.rect(self.screen, (40, 40, 60), (10, 10, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 10, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)

        # --- Connections Text ---
        conn_text = f"CONNECTIONS: {self.total_connections} / {self.NUM_PAIRS}"
        text_surf = self.font_ui.render(conn_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
    
    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "VICTORY" if self.total_connections >= self.NUM_PAIRS else "OUT OF MOMENTUM"
        text_surf = self.font_big.render(message, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    # --- Particle System ---
    def _spawn_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # drag
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30.0))
            color = (*p["color"], alpha)
            size = int(p["lifetime"] / 10) + 1
            pygame.draw.circle(self.screen, color, p["pos"].astype(int), size)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self._active_puzzle["momentum"],
            "connections": self.total_connections,
            "recursion_level": self._active_puzzle["level"]
        }

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure we have a display for human play
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Node Connector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        # --- Human Input to Action Mapping ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        clock.tick(GameEnv.FPS)

    env.close()