import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a digital network by teleporting between nodes. "
        "Evade security probes and disable firewalls to access the entire system."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport between nodes. "
        "Hold SPACE to activate time dilation and slow down security."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_NODE_UNACCESSED = (70, 70, 90)
    COLOR_NODE_ACCESSED = (0, 255, 100)
    COLOR_NODE_UPGRADED = (170, 0, 255)
    COLOR_CONNECTION = (40, 50, 80)
    COLOR_PROBE = (255, 50, 50)
    COLOR_PROBE_GLOW = (128, 25, 25)
    COLOR_FIREWALL_WARN = (255, 255, 0)
    COLOR_FIREWALL_ACTIVE = (255, 0, 0)
    COLOR_TIME_DILATION_EFFECT = (255, 220, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game parameters
    MAX_STEPS = 2000
    NUM_NODES = 15
    NUM_PROBES = 3
    MIN_NODE_DISTANCE = 80

    PLAYER_SPEED = 0.1  # Teleport progress per step
    PROBE_BASE_SPEED = 0.005
    PROBE_SPEED_INCREASE_INTERVAL = 200
    PROBE_SPEED_INCREASE_AMOUNT = 0.005

    FIREWALL_BASE_FREQ = 0.002
    FIREWALL_FREQ_INCREASE_INTERVAL = 100
    FIREWALL_FREQ_INCREASE_PERCENT = 0.01

    TIME_DILATION_FACTOR = 0.25
    TIME_DILATION_FUEL_MAX = 100
    TIME_DILATION_FUEL_DRAIN = 2.0
    TIME_DILATION_FUEL_REGEN = 0.5

    UPGRADE_NODES_TIME = 5
    UPGRADE_NODES_TELEPORT = 10

    # --- Helper Classes ---
    class Node:
        def __init__(self, idx, pos):
            self.id = idx
            self.pos = pygame.Vector2(pos)
            self.connections = []
            self.accessed = False
            self.is_upgrade_node = False

            self.firewall_state = "off"  # "off", "warning", "on"
            self.firewall_timer = random.uniform(0, 1000)
            self.firewall_cycle_time = random.uniform(200, 300)

    class SecurityProbe:
        def __init__(self, path):
            self.path = path  # List of node indices
            self.current_path_idx = 0
            self.lerp_t = 0.0
            self.speed = GameEnv.PROBE_BASE_SPEED
            self.pos = pygame.Vector2(0, 0)

    class Particle:
        def __init__(self, pos, vel, size, lifespan, color):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.size = size
            self.lifespan = lifespan
            self.life = lifespan
            self.color = color

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.nodes = []
        self.probes = []
        self.particles = []

        self.player_node_idx = 0
        self.is_teleporting = False
        self.teleport_origin_idx = 0
        self.teleport_target_idx = 0
        self.teleport_progress = 0.0
        self.player_pos = pygame.Vector2(0, 0)

        self.time_dilation_active = False
        self.time_dilation_fuel = self.TIME_DILATION_FUEL_MAX
        self.time_dilation_duration_upgrade = False

        self.teleport_range_upgrade = False

        self.steps = 0
        self.score = 0
        self.accessed_node_count = 0
        self.game_over = False
        self.game_won = False

        self._generate_level()

    def _generate_level(self):
        # Generate nodes
        self.nodes = []
        padding = 50
        attempts = 0
        while len(self.nodes) < self.NUM_NODES and attempts < 1000:
            attempts += 1
            pos = (
                random.randint(padding, self.SCREEN_WIDTH - padding),
                random.randint(padding, self.SCREEN_HEIGHT - padding),
            )

            too_close = False
            for node in self.nodes:
                if node.pos.distance_to(pos) < self.MIN_NODE_DISTANCE:
                    too_close = True
                    break
            if not too_close:
                self.nodes.append(self.Node(len(self.nodes), pos))

        # Connect nodes using a minimum spanning tree + extra edges
        edges = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = self.nodes[i].pos.distance_to(self.nodes[j].pos)
                edges.append((dist, i, j))
        edges.sort()

        parent = list(range(len(self.nodes)))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                return True
            return False

        for dist, i, j in edges:
            if union(i, j):
                self.nodes[i].connections.append(j)
                self.nodes[j].connections.append(i)

        # Add a few extra edges for more paths
        extra_edges = int(len(self.nodes) * 0.5)
        for dist, i, j in edges:
            if extra_edges <= 0:
                break
            if j not in self.nodes[i].connections:
                self.nodes[i].connections.append(j)
                self.nodes[j].connections.append(i)
                extra_edges -= 1

        # Designate upgrade nodes
        node_ids = list(range(len(self.nodes)))
        random.shuffle(node_ids)
        if len(node_ids) > 0:
            self.nodes[node_ids.pop()].is_upgrade_node = True
        if len(node_ids) > 0:
            self.nodes[node_ids.pop()].is_upgrade_node = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_level()

        self.steps = 0
        self.score = 0
        self.accessed_node_count = 0
        self.game_over = False
        self.game_won = False

        self.player_node_idx = random.randint(0, len(self.nodes) - 1)
        self.nodes[self.player_node_idx].accessed = True
        self.accessed_node_count = 1
        self.player_pos = self.nodes[self.player_node_idx].pos.copy()

        self.is_teleporting = False
        self.teleport_progress = 0.0

        self.time_dilation_active = False
        self.time_dilation_fuel = self.TIME_DILATION_FUEL_MAX
        self.time_dilation_duration_upgrade = False
        self.teleport_range_upgrade = False

        self.particles.clear()

        # Reset firewalls
        for node in self.nodes:
            node.accessed = False
            node.firewall_state = "off"
            node.firewall_timer = random.uniform(0, 1000)
        self.nodes[self.player_node_idx].accessed = True

        # Generate probes
        self.probes = []
        for _ in range(self.NUM_PROBES):
            path = [random.randint(0, len(self.nodes) - 1)]
            for _ in range(random.randint(2, 4)):
                last_node = self.nodes[path[-1]]
                if last_node.connections:
                    path.append(random.choice(last_node.connections))
            self.probes.append(self.SecurityProbe(path))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Update Game Logic ---
        self.steps += 1

        # 1. Handle time dilation
        if space_held and self.time_dilation_fuel > 0:
            self.time_dilation_active = True
            fuel_drain_multiplier = (
                0.5 if self.time_dilation_duration_upgrade else 1.0
            )
            self.time_dilation_fuel = max(
                0, self.time_dilation_fuel - self.TIME_DILATION_FUEL_DRAIN * fuel_drain_multiplier
            )
        else:
            self.time_dilation_active = False
            self.time_dilation_fuel = min(
                self.TIME_DILATION_FUEL_MAX,
                self.time_dilation_fuel + self.TIME_DILATION_FUEL_REGEN,
            )
            if self.time_dilation_fuel <= 0:
                self.time_dilation_active = False

        time_multiplier = self.TIME_DILATION_FACTOR if self.time_dilation_active else 1.0

        # 2. Handle player teleportation
        if not self.is_teleporting and movement != 0:
            self._handle_teleport_action(movement)

        if self.is_teleporting:
            self.teleport_progress += self.PLAYER_SPEED
            origin_pos = self.nodes[self.teleport_origin_idx].pos
            target_pos = self.nodes[self.teleport_target_idx].pos
            self.player_pos = origin_pos.lerp(target_pos, min(1.0, self.teleport_progress))

            # Teleport trail
            if self.steps % 2 == 0:
                self.particles.append(
                    self.Particle(self.player_pos, (0, 0), 8, 20, self.COLOR_PLAYER)
                )

            if self.teleport_progress >= 1.0:
                self.is_teleporting = False
                self.player_node_idx = self.teleport_target_idx
                self.player_pos = self.nodes[self.player_node_idx].pos.copy()

                # Check for new node access
                if not self.nodes[self.player_node_idx].accessed:
                    self.nodes[self.player_node_idx].accessed = True
                    self.accessed_node_count += 1
                    reward += 1.0

                    # Check for upgrades
                    if (
                        self.accessed_node_count >= self.UPGRADE_NODES_TIME
                        and not self.time_dilation_duration_upgrade
                    ):
                        self.time_dilation_duration_upgrade = True
                        reward += 5.0
                    if (
                        self.accessed_node_count >= self.UPGRADE_NODES_TELEPORT
                        and not self.teleport_range_upgrade
                    ):
                        self.teleport_range_upgrade = True
                        reward += 5.0

                    if self.nodes[self.player_node_idx].is_upgrade_node:
                        self.nodes[self.player_node_idx].is_upgrade_node = False
                        self.score += 50
                        reward += 2.0

        # 3. Update firewalls
        for node in self.nodes:
            if self.time_dilation_active:
                node.firewall_state = "off"
                continue

            node.firewall_timer += time_multiplier
            if node.firewall_timer > node.firewall_cycle_time:
                node.firewall_timer = 0
                node.firewall_state = "warning"

            if node.firewall_state == "warning" and node.firewall_timer > 50:
                node.firewall_state = "on"
            elif node.firewall_state == "on" and node.firewall_timer > 100:
                node.firewall_state = "off"

        # 4. Update security probes
        speed_increase = (
            self.steps // self.PROBE_SPEED_INCREASE_INTERVAL
        ) * self.PROBE_SPEED_INCREASE_AMOUNT
        for probe in self.probes:
            probe.speed = self.PROBE_BASE_SPEED + speed_increase
            probe.lerp_t += probe.speed * time_multiplier

            start_node_idx = probe.path[probe.current_path_idx]
            end_node_idx = probe.path[(probe.current_path_idx + 1) % len(probe.path)]

            probe.pos = self.nodes[start_node_idx].pos.lerp(
                self.nodes[end_node_idx].pos, min(1.0, probe.lerp_t)
            )

            if probe.lerp_t >= 1.0:
                probe.lerp_t = 0.0
                probe.current_path_idx = (probe.current_path_idx + 1) % len(probe.path)

        # 5. Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.pos += p.vel
            p.life -= 1
            p.size = max(0, p.size * (p.life / p.lifespan))

        # 6. Check for termination conditions
        # Detection by probe
        for probe in self.probes:
            if probe.pos.distance_to(self.player_pos) < 15:
                self.game_over = True
                reward = -100.0
                self._create_explosion(self.player_pos, self.COLOR_PROBE)
                break

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        # Win condition
        if self.accessed_node_count == len(self.nodes):
            self.game_over = True
            self.game_won = True
            reward = 50.0
            self._create_explosion(self.player_pos, self.COLOR_NODE_ACCESSED)

        # 7. Calculate final reward for the step
        if not self.game_over:
            reward += 0.01  # Small survival reward

        self.score += reward

        return self._get_observation(), reward, self.game_over, self.steps >= self.MAX_STEPS, self._get_info()

    def _handle_teleport_action(self, movement):
        direction_vectors = {
            1: pygame.Vector2(0, -1),  # Up
            2: pygame.Vector2(0, 1),  # Down
            3: pygame.Vector2(-1, 0),  # Left
            4: pygame.Vector2(1, 0),  # Right
        }

        move_dir = direction_vectors.get(movement)
        if not move_dir:
            return

        current_node = self.nodes[self.player_node_idx]
        best_target = -1
        max_dot = -1.1  # Must be better than -1

        # Determine potential targets
        potential_targets = set(current_node.connections)
        if self.teleport_range_upgrade:
            # Add connections of connections (2-hop neighbors)
            for conn_idx in current_node.connections:
                for second_conn_idx in self.nodes[conn_idx].connections:
                    if second_conn_idx != self.player_node_idx:
                        potential_targets.add(second_conn_idx)

        for target_idx in potential_targets:
            target_node = self.nodes[target_idx]

            # Check if firewall is active
            if target_node.firewall_state == "on":
                continue

            vec_to_target = (target_node.pos - current_node.pos).normalize()
            dot_product = move_dir.dot(vec_to_target)

            if dot_product > max_dot:
                max_dot = dot_product
                best_target = target_idx

        # Threshold to prevent teleporting backwards
        if best_target != -1 and max_dot > 0.3:
            self.is_teleporting = True
            self.teleport_origin_idx = self.player_node_idx
            self.teleport_target_idx = best_target
            self.teleport_progress = 0.0

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
            "accessed_nodes": self.accessed_node_count,
            "time_dilation_fuel": self.time_dilation_fuel,
            "upgrades": {
                "time_dilation": self.time_dilation_duration_upgrade,
                "teleport_range": self.teleport_range_upgrade,
            },
        }

    def _render_game(self):
        # Grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Time dilation effect
        if self.time_dilation_active:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((*self.COLOR_TIME_DILATION_EFFECT, 30))
            self.screen.blit(s, (0, 0))

        # Connections
        for node in self.nodes:
            for conn_idx in node.connections:
                if conn_idx > node.id:  # Draw each connection once
                    pygame.draw.line(
                        self.screen, self.COLOR_CONNECTION, node.pos, self.nodes[conn_idx].pos, 2
                    )

        # Nodes and Firewalls
        for node in self.nodes:
            color = self.COLOR_NODE_UNACCESSED
            if node.accessed:
                color = self.COLOR_NODE_ACCESSED
            if node.is_upgrade_node:
                color = self.COLOR_NODE_UPGRADED

            self._draw_glow_circle(self.screen, color, node.pos, 8, 32)
            pygame.gfxdraw.filled_circle(
                self.screen, int(node.pos.x), int(node.pos.y), 6, color
            )

            # Firewall visuals
            if node.firewall_state == "warning":
                pulse = abs(math.sin(self.steps * 0.2))
                pygame.gfxdraw.aacircle(
                    self.screen,
                    int(node.pos.x),
                    int(node.pos.y),
                    int(12 + pulse * 4),
                    self.COLOR_FIREWALL_WARN,
                )
            elif node.firewall_state == "on":
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(node.pos.x),
                    int(node.pos.y),
                    15,
                    (*self.COLOR_FIREWALL_ACTIVE, 100),
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(node.pos.x), int(node.pos.y), 15, self.COLOR_FIREWALL_ACTIVE
                )

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.lifespan))
            color = (*p.color, alpha)
            s = pygame.Surface((p.size * 2, p.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size, p.size), p.size)
            self.screen.blit(
                s, p.pos - pygame.Vector2(p.size, p.size), special_flags=pygame.BLEND_RGBA_ADD
            )

        # Teleport streak
        if self.is_teleporting:
            start = self.nodes[self.teleport_origin_idx].pos
            pygame.draw.line(self.screen, self.COLOR_PLAYER, start, self.player_pos, 3)

        # Probes
        for probe in self.probes:
            self._draw_glow_circle(self.screen, self.COLOR_PROBE_GLOW, probe.pos, 12, 64)
            p1 = probe.pos + pygame.Vector2(0, -8)
            p2 = probe.pos + pygame.Vector2(-7, 6)
            p3 = probe.pos + pygame.Vector2(7, 6)
            pygame.gfxdraw.filled_trigon(
                self.screen,
                int(p1.x),
                int(p1.y),
                int(p2.x),
                int(p2.y),
                int(p3.x),
                int(p3.y),
                self.COLOR_PROBE,
            )

        # Player
        if not self.game_over:
            size = 12
            glow_size = 24
            player_rect = pygame.Rect(
                self.player_pos.x - size / 2, self.player_pos.y - size / 2, size, size
            )
            glow_rect = pygame.Rect(
                self.player_pos.x - glow_size / 2,
                self.player_pos.y - glow_size / 2,
                glow_size,
                glow_size,
            )

            s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLAYER_GLOW, 128), s.get_rect(), border_radius=4)
            self.screen.blit(s, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Access Level
        access_text = self.font_ui.render(
            f"ACCESS: {self.accessed_node_count}/{len(self.nodes)}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(access_text, (10, 10))

        # Time Dilation Bar
        bar_w, bar_h = 150, 15
        bar_x, bar_y = self.SCREEN_WIDTH - bar_w - 10, 10
        fuel_ratio = self.time_dilation_fuel / self.TIME_DILATION_FUEL_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        fill_color = (
            self.COLOR_TIME_DILATION_EFFECT
            if self.time_dilation_active or fuel_ratio > 0.1
            else self.COLOR_PROBE
        )
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_w * fuel_ratio, bar_h))

        # Upgrades
        y_offset = self.SCREEN_HEIGHT - 30
        if self.time_dilation_duration_upgrade:
            upgrade_text = self.font_ui.render("TIME++", True, self.COLOR_NODE_UPGRADED)
            self.screen.blit(upgrade_text, (10, y_offset))
            y_offset -= 25
        if self.teleport_range_upgrade:
            upgrade_text = self.font_ui.render("TELEPORT++", True, self.COLOR_NODE_UPGRADED)
            self.screen.blit(upgrade_text, (10, y_offset))

        # Game Over/Win Text
        if self.game_over:
            text_str = "SYSTEM CRASH"
            text_color = self.COLOR_PROBE
            if self.game_won:
                text_str = "SYSTEM MASTERED"
                text_color = self.COLOR_NODE_ACCESSED

            text_surface = self.font_game_over.render(text_str, True, text_color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _draw_glow_circle(self, surface, color, center, radius, max_alpha):
        for i in range(4):
            alpha = max_alpha * (1 - i / 4)
            rad = radius + i * 2
            s = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (rad, rad), rad)
            surface.blit(
                s, (center[0] - rad, center[1] - rad), special_flags=pygame.BLEND_RGBA_ADD
            )

    def _create_explosion(self, pos, color):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = random.uniform(2, 8)
            lifespan = random.randint(20, 40)
            self.particles.append(self.Particle(pos, vel, size, lifespan, color))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually.
    # It will not be run by the evaluation server.
    #
    # To use, you will need to `pip install pygame`.
    #
    # Controls:
    # - Arrow Keys: Teleport
    # - Space: Activate Time Dilation
    # - R: Reset game
    
    # Un-comment the line below to run with a graphical display
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("CyberNode Infiltrator")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_held = 0  # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            if keys[pygame.K_SPACE]:
                space_held = 1

            action = [movement, space_held, 0]  # Shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)  # Run at 30 FPS

    env.close()