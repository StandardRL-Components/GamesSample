import gymnasium as gym
import os
import math
import random
import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend the mother tree by expanding your root network to repel encroaching vines. Manage nutrients and create defensive pulses to survive."
    user_guide = "Use arrow keys to select nodes, space to connect them, and shift to boost nutrient flow to the selected node."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1200  # Increased for longer gameplay
    WIN_BONUS = 100.0
    SURVIVAL_REWARD = 0.01
    VINE_REPEL_REWARD = 1.0

    # --- Colors ---
    COLOR_BG = pygame.Color(10, 20, 15)
    COLOR_TREE = pygame.Color(101, 67, 33)
    COLOR_ROOT_DEAD = pygame.Color(60, 40, 30)
    COLOR_ROOT_HEALTHY = pygame.Color(144, 238, 144)
    COLOR_NUTRIENT = pygame.Color(173, 216, 230)
    COLOR_VINE = pygame.Color(220, 20, 60)
    COLOR_VINE_GLOW = pygame.Color(255, 105, 180)
    COLOR_UI_TEXT = pygame.Color(240, 240, 240)
    COLOR_HEALTH_BAR = pygame.Color(40, 200, 40)
    COLOR_HEALTH_BAR_BG = pygame.Color(80, 0, 0)
    COLOR_NODE = pygame.Color(200, 200, 200)
    COLOR_NODE_SELECTED = pygame.Color(255, 255, 0)
    COLOR_NODE_PREVIOUS = pygame.Color(255, 165, 0)
    COLOR_PULSE = pygame.Color(255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables are initialized in reset()
        self.nodes = []
        self.connections = {}
        self.vines = []
        self.particles = []
        self.effects = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.tree_health = 100.0
        self.vine_growth_rate = 0.15
        self.base_health_decay = 0.02

        self.selected_node_idx = 0
        self.previous_selected_node_idx = None

        self.selection_interp_pos = None
        self.selection_interp_timer = 0

        # --- Game World Setup ---
        self.nodes = []
        self.connections = {}
        self.vines = []
        self.particles = []
        self.effects = []

        # Create the central source node (part of the tree)
        self.nodes.append(
            {
                "id": 0,
                "pos": (self.WIDTH // 2, self.HEIGHT - 50),
                "potential": 1.0,
                "is_source": True,
            }
        )

        # Create a ring of potential nodes
        num_nodes = 16
        radius_x, radius_y = 250, 150
        for i in range(1, num_nodes + 1):
            angle = (i / num_nodes) * 2 * math.pi
            x = self.WIDTH // 2 + math.cos(angle) * (
                radius_x + self.np_random.uniform(-30, 30)
            )
            y = (self.HEIGHT - 80) - math.sin(angle) * (
                radius_y + self.np_random.uniform(-30, 30)
            )
            self.nodes.append(
                {"id": i, "pos": (x, y), "potential": 0.0, "is_source": False}
            )

        # Spawn initial vines
        for _ in range(8):
            self._spawn_vine()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1

        reward = 0

        # --- Handle Player Actions ---
        self._handle_movement(movement)
        self._handle_connection(space_pressed)
        self._handle_boost(shift_held)

        # --- Update Game Logic ---
        self._update_difficulty()
        self._update_nutrients()
        self._update_vines()

        repel_reward = self._update_defense()
        reward += repel_reward

        self._update_tree_health()
        self._update_particles()
        self._update_effects()

        self.steps += 1
        self.score += repel_reward

        # --- Calculate Reward and Termination ---
        terminated = self.tree_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # Per Gymnasium API, this is for time limits not part of the MDP
        if not terminated:
            reward += self.SURVIVAL_REWARD
        elif self.steps >= self.MAX_STEPS and self.tree_health > 0:
            reward += self.WIN_BONUS
            self.score += self.WIN_BONUS

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    # --- Internal Logic Methods ---

    def _spawn_vine(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0:  # Top
            pos = [self.np_random.uniform(0, self.WIDTH), 0]
        elif edge == 1:  # Bottom
            pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT]
        elif edge == 2:  # Left
            pos = [0, self.np_random.uniform(0, self.HEIGHT)]
        else:  # Right
            pos = [self.WIDTH, self.np_random.uniform(0, self.HEIGHT)]
        self.vines.append({"segments": [pos], "stopped": False})

    def _handle_movement(self, movement):
        if movement == 0:  # No-op
            return

        current_node = self.nodes[self.selected_node_idx]
        current_pos = current_node["pos"]

        candidates = []
        for i, node in enumerate(self.nodes):
            if i == self.selected_node_idx:
                continue

            dx = node["pos"][0] - current_pos[0]
            dy = node["pos"][1] - current_pos[1]

            if dx == 0 and dy == 0:
                continue

            is_candidate = False
            if movement == 1:  # Up
                if dy < 0 and abs(dy) > abs(dx):
                    is_candidate = True
            elif movement == 2:  # Down
                if dy > 0 and abs(dy) > abs(dx):
                    is_candidate = True
            elif movement == 3:  # Left
                if dx < 0 and abs(dx) > abs(dy):
                    is_candidate = True
            elif movement == 4:  # Right
                if dx > 0 and abs(dx) > abs(dy):
                    is_candidate = True

            if is_candidate:
                dist_sq = dx**2 + dy**2
                candidates.append((dist_sq, i))

        if candidates:
            # Sfx: UI_SELECT
            _, best_idx = min(candidates, key=lambda x: x[0])
            if self.selected_node_idx != best_idx:
                self.previous_selected_node_idx = self.selected_node_idx
                self.selected_node_idx = best_idx
                self.selection_interp_pos = self.nodes[
                    self.previous_selected_node_idx
                ]["pos"]
                self.selection_interp_timer = 5  # frames

    def _handle_connection(self, space_pressed):
        if not space_pressed or self.previous_selected_node_idx is None:
            return

        u, v = self.selected_node_idx, self.previous_selected_node_idx
        if u == v:
            return

        # Ensure consistent key order
        key = tuple(sorted((u, v)))
        if key not in self.connections:
            # Sfx: ROOT_CONNECT
            pos1 = self.nodes[key[0]]["pos"]
            pos2 = self.nodes[key[1]]["pos"]
            length = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
            self.connections[key] = {
                "nutrient_level": 0.0,
                "length": length,
                "last_pulse": -100,
            }
            self.effects.append(
                {
                    "type": "pulse",
                    "pos": ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2),
                    "radius": 10,
                    "max_radius": 20,
                    "life": 10,
                    "color": self.COLOR_NODE_SELECTED,
                }
            )

    def _handle_boost(self, shift_held):
        for node in self.nodes:
            node["boosted"] = False
        if shift_held:
            # Sfx: BOOST_CHARGE
            self.nodes[self.selected_node_idx]["boosted"] = True

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.vine_growth_rate += 0.01
            if len(self.vines) < 20:  # Cap number of vines
                self._spawn_vine()

    def _update_nutrients(self):
        # Reset potentials
        for node in self.nodes:
            if not node["is_source"]:
                node["potential"] = 0.0

        # Propagate potential (simple iterative approach)
        for _ in range(5):  # Iterations for stability
            for u, v in self.connections.keys():
                n1, n2 = self.nodes[u], self.nodes[v]
                # Potential flows from higher to lower
                p1, p2 = n1["potential"], n2["potential"]
                n1["potential"] = max(p1, p2 * 0.9)
                n2["potential"] = max(p2, p1 * 0.9)

        # Update nutrient levels in connections based on potential
        for key, conn in self.connections.items():
            u, v = key
            n1, n2 = self.nodes[u], self.nodes[v]

            # Boosted nodes act as a sink, drawing nutrients
            boost_factor = 1.0
            if n1.get("boosted", False) or n2.get("boosted", False):
                boost_factor = 5.0

            potential_diff = abs(n1["potential"] - n2["potential"])
            flow = potential_diff * 0.1 * boost_factor
            conn["nutrient_level"] = min(100.0, conn["nutrient_level"] + flow)

            # Spawn particles based on flow
            if self.np_random.random() < flow * 0.5:
                start_pos = (
                    n1["pos"] if n1["potential"] > n2["potential"] else n2["pos"]
                )
                end_pos = n2["pos"] if n1["potential"] > n2["potential"] else n1["pos"]
                self.particles.append(
                    {
                        "pos": list(start_pos),
                        "end_pos": end_pos,
                        "life": conn["length"] / 3,
                        "max_life": conn["length"] / 3,
                    }
                )

    def _update_vines(self):
        tree_center = (self.WIDTH // 2, self.HEIGHT - 40)
        tree_radius_sq = 20**2

        for vine in self.vines:
            if vine["stopped"]:
                continue

            last_segment = vine["segments"][-1]

            # Grow towards the tree center
            angle_to_center = math.atan2(
                tree_center[1] - last_segment[1], tree_center[0] - last_segment[0]
            )

            # Add some organic waviness
            angle_to_center += math.sin(self.steps * 0.1 + id(vine) % 100) * 0.5

            dx = math.cos(angle_to_center) * self.vine_growth_rate
            dy = math.sin(angle_to_center) * self.vine_growth_rate

            new_segment = [last_segment[0] + dx, last_segment[1] + dy]
            vine["segments"].append(new_segment)

            # Prune very long vines to save memory
            if len(vine["segments"]) > 200:
                vine["segments"].pop(0)

            # Check for collision with tree
            dist_sq = (new_segment[0] - tree_center[0]) ** 2 + (
                new_segment[1] - tree_center[1]
            ) ** 2
            if dist_sq < tree_radius_sq:
                vine["stopped"] = True
                self.tree_health -= 5.0  # Significant damage on contact
                # Sfx: TREE_DAMAGE
                self.effects.append(
                    {
                        "type": "pulse",
                        "pos": tree_center,
                        "radius": 20,
                        "max_radius": 40,
                        "life": 15,
                        "color": self.COLOR_VINE,
                    }
                )

    def _update_defense(self):
        reward = 0
        for key, conn in self.connections.items():
            if conn["nutrient_level"] >= 100.0 and (self.steps - conn["last_pulse"]) > 60:
                # Sfx: PULSE_DEFENSE
                conn["nutrient_level"] = 0.0
                conn["last_pulse"] = self.steps

                n1, n2 = self.nodes[key[0]], self.nodes[key[1]]
                mid_point = (
                    (n1["pos"][0] + n2["pos"][0]) / 2,
                    (n1["pos"][1] + n2["pos"][1]) / 2,
                )
                pulse_radius = 60

                self.effects.append(
                    {
                        "type": "pulse",
                        "pos": mid_point,
                        "radius": 10,
                        "max_radius": pulse_radius,
                        "life": 20,
                        "color": self.COLOR_PULSE,
                    }
                )

                # Check for vines to repel
                for vine in self.vines:
                    initial_len = len(vine["segments"])

                    # Keep segments that are far from the pulse
                    vine["segments"] = [
                        s
                        for s in vine["segments"]
                        if math.hypot(s[0] - mid_point[0], s[1] - mid_point[1])
                        > pulse_radius
                    ]

                    if not vine["segments"]:  # If all segments are destroyed
                        vine["stopped"] = True  # effectively remove it

                    if len(vine["segments"]) < initial_len and vine["stopped"]:
                        vine["stopped"] = False  # Revive vine if it was stopped at the tree

                    num_destroyed = initial_len - len(vine["segments"])
                    if num_destroyed > 0:
                        reward += self.VINE_REPEL_REWARD

        return reward

    def _update_tree_health(self):
        vines_on_tree = sum(1 for v in self.vines if v["stopped"])
        damage = self.base_health_decay + (vines_on_tree * 0.1)
        self.tree_health = max(0, self.tree_health - damage)

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue

            # Move particle towards its destination
            dx = p["end_pos"][0] - p["pos"][0]
            dy = p["end_pos"][1] - p["pos"][1]
            p["pos"][0] += dx * 0.1
            p["pos"][1] += dy * 0.1

    def _update_effects(self):
        for e in self.effects[:]:
            e["life"] -= 1
            if e["life"] <= 0:
                self.effects.remove(e)
            else:
                e["radius"] += (e["max_radius"] - e["radius"]) * 0.2

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_vines()
        self._render_connections()
        self._render_particles()
        self._render_nodes()
        self._render_tree()
        self._render_effects()
        self._render_selection_highlight()

    def _render_tree(self):
        health_ratio = self.tree_health / 100.0
        trunk_width = int(15 + 15 * health_ratio)
        trunk_height = int(40 + 20 * health_ratio)
        trunk_color = self.COLOR_TREE

        base_pos = self.nodes[0]["pos"]
        trunk_rect = pygame.Rect(
            base_pos[0] - trunk_width // 2,
            base_pos[1] - trunk_height,
            trunk_width,
            trunk_height,
        )
        pygame.draw.rect(
            self.screen,
            trunk_color,
            trunk_rect,
            border_top_left_radius=5,
            border_top_right_radius=5,
        )

        # Draw some ground
        pygame.draw.circle(self.screen, self.COLOR_TREE, (base_pos[0], base_pos[1] + 5), 30)

    def _render_connections(self):
        for key, conn in self.connections.items():
            n1, n2 = self.nodes[key[0]], self.nodes[key[1]]
            p1, p2 = n1["pos"], n2["pos"]

            nutrient_ratio = conn["nutrient_level"] / 100.0
            color = self.COLOR_ROOT_DEAD.lerp(self.COLOR_ROOT_HEALTHY, nutrient_ratio)
            width = int(1 + 3 * nutrient_ratio)

            pygame.draw.line(self.screen, color, p1, p2, width)

    def _render_nodes(self):
        for i, node in enumerate(self.nodes):
            pos = (int(node["pos"][0]), int(node["pos"][1]))
            color = self.COLOR_NODE
            radius = 5

            if node.get("boosted", False):
                # Draw a pulsing glow for boosted nodes
                pulse_size = 10 + 3 * math.sin(self.steps * 0.2)
                color_with_alpha = (
                    self.COLOR_NUTRIENT.r,
                    self.COLOR_NUTRIENT.g,
                    self.COLOR_NUTRIENT.b,
                    50,
                )
                pygame.gfxdraw.filled_circle(
                    self.screen, pos[0], pos[1], int(pulse_size), color_with_alpha
                )

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_selection_highlight(self):
        # Interpolate selection cursor for smooth movement
        if self.selection_interp_timer > 0:
            self.selection_interp_timer -= 1
            target_pos = self.nodes[self.selected_node_idx]["pos"]
            dx = target_pos[0] - self.selection_interp_pos[0]
            dy = target_pos[1] - self.selection_interp_pos[1]
            self.selection_interp_pos = (
                self.selection_interp_pos[0] + dx * 0.4,
                self.selection_interp_pos[1] + dy * 0.4,
            )
            pos = (int(self.selection_interp_pos[0]), int(self.selection_interp_pos[1]))
        else:
            pos = self.nodes[self.selected_node_idx]["pos"]

        # Draw main selection
        pygame.gfxdraw.aacircle(
            self.screen, int(pos[0]), int(pos[1]), 10, self.COLOR_NODE_SELECTED
        )

        # Draw previous selection
        if self.previous_selected_node_idx is not None:
            prev_pos = self.nodes[self.previous_selected_node_idx]["pos"]
            pygame.gfxdraw.aacircle(
                self.screen, int(prev_pos[0]), int(prev_pos[1]), 8, self.COLOR_NODE_PREVIOUS
            )

    def _render_vines(self):
        for vine in self.vines:
            if len(vine["segments"]) < 2:
                continue

            # Glow effect
            pygame.draw.lines(
                self.screen, self.COLOR_VINE_GLOW, False, vine["segments"], width=7
            )
            # Core vine
            pygame.draw.lines(
                self.screen, self.COLOR_VINE, False, vine["segments"], width=3
            )

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, self.COLOR_NUTRIENT)

    def _render_effects(self):
        for e in self.effects:
            if e["type"] == "pulse":
                alpha = int(200 * (e["life"] / 20)) if e["life"] > 0 else 0
                if alpha > 0:
                    color = (*e["color"][:3], alpha)
                    pygame.gfxdraw.aacircle(
                        self.screen, int(e["pos"][0]), int(e["pos"][1]), int(e["radius"]), color
                    )

    def _render_ui(self):
        # Health Bar
        health_percent = self.tree_health / 100.0
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(
            self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height)
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_HEALTH_BAR,
            (10, 10, bar_width * health_percent, bar_height),
        )

        # Score and Time
        score_text = self.font_small.render(
            f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        time_text = self.font_small.render(
            f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 30))

        if self.game_over:
            msg = "VICTORY" if self.tree_health > 0 else "DEFEAT"
            color = self.COLOR_HEALTH_BAR if self.tree_health > 0 else self.COLOR_VINE
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tree_health": self.tree_health,
            "vines": len(self.vines),
            "connections": len(self.connections),
        }

    def close(self):
        pygame.quit()


# Example usage to run and visualize the environment
if __name__ == "__main__":
    # This block will not be run by the autograder, but is useful for testing
    # Note: You may need to unset SDL_VIDEODRIVER for this to work on your machine
    # For example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")

    # To play manually, map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Root Network Defender")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False

    print("\n--- Controls ---")
    print("Arrow Keys: Select node")
    print("Spacebar: Connect selected to previously selected node")
    print("Shift: Boost nutrient flow to selected node")
    print("R: Reset environment")
    print("----------------\n")

    while True:
        # --- Human Input ---
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break

        if keys[pygame.K_SPACE]:
            space_action = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered image
        # We just need to get it on the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)