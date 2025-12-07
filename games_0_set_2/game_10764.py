import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:54:23.461637
# Source Brief: brief_00764.md
# Brief Index: 764
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a dynamic cybernetic grid, using special abilities to maintain momentum and reach the exit before your connection is severed by security systems."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the grid. Press space and shift to activate your equipped cybernetic cards."
    )
    auto_advance = True

    # --- Persistent State for Unlocks ---
    AVAILABLE_CARDS = [
        {"name": "Momentum Burst", "color": (255, 255, 0), "effect": "add_momentum", "value": 30, "cooldown": 200},
        {"name": "Shield", "color": (0, 255, 255), "effect": "shield", "value": 150, "cooldown": 300}, # 5 seconds at 30fps
        {"name": "Pathfinder", "color": (255, 0, 255), "effect": "pathfind", "value": 210, "cooldown": 400}, # 7 seconds
    ]
    unlocked_card_indices = {0}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Visual & Game Feel Constants ---
        self.FPS = 30
        self.GRID_SIZE = 40
        self.PLAYER_SPEED = 4.0

        # --- Colors (Bright on Dark) ---
        self.COLOR_BG = (5, 0, 15)
        self.COLOR_GRID = (20, 30, 80)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_TRAIL = (255, 255, 0)
        self.COLOR_EXIT = (170, 0, 255)
        self.COLOR_SECURITY = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_MOMENTUM_BAR = (255, 190, 0)
        
        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.reward_this_step = None
        
        self.network = None
        self.grid_w = None
        self.grid_h = None

        self.player_pos = None
        self.player_grid_pos = None
        self.player_target_pos = None
        self.is_moving = None
        self.momentum = None
        
        self.exit_pos = None
        self.security_nodes = None
        self.base_pulse_interval = None
        self.current_pulse_interval = None

        self.camera_pos = None
        self.particles = None
        self.trail_particles = None

        self.player_hand = None
        self.card_cooldowns = None
        self.active_effects = None

        self.prev_space_held = None
        self.prev_shift_held = None

        self.shortest_path = None

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # --- Network Generation ---
        self._generate_network()
        start_pos = (self.np_random.integers(1, self.grid_w-1), self.np_random.integers(1, self.grid_h-1))
        
        # --- Player State ---
        self.player_grid_pos = start_pos
        self.player_pos = pygame.Vector2(start_pos) * self.GRID_SIZE
        self.player_target_pos = self.player_pos.copy()
        self.is_moving = False
        self.momentum = 100.0

        # --- Camera ---
        self.camera_pos = self.player_pos.copy()

        # --- Game Elements ---
        self._place_exit_node(start_pos)
        self.base_pulse_interval = 2.0 * self.FPS # 2 seconds
        self.current_pulse_interval = self.base_pulse_interval
        self._place_security_nodes(start_pos, self.exit_pos)

        # --- Effects & UI ---
        self.particles = []
        self.trail_particles = deque(maxlen=20)
        self.active_effects = {"shield": 0, "pathfind": 0}
        self._draw_hand()
        self.prev_space_held = False
        self.prev_shift_held = False
        self.shortest_path = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        # --- Handle Input & State Updates ---
        self._handle_input(action)
        self._update_player()
        self._update_security_nodes()
        self._update_effects_and_cooldowns()
        
        # --- Rewards & Termination ---
        self._calculate_reward()
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated and not self.game_over:
            if self.player_grid_pos == self.exit_pos:
                # Victory
                self.reward_this_step += 100
                self.score += 100
                self._unlock_new_card()
                # sfx: game_win
            else:
                # Failure
                self.reward_this_step -= 100
                self.score -= 100
                # sfx: game_over
            self.game_over = True

        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # --- Movement ---
        if not self.is_moving:
            move_dir = None
            if movement == 1: move_dir = (0, -1) # Up
            elif movement == 2: move_dir = (0, 1)  # Down
            elif movement == 3: move_dir = (-1, 0) # Left
            elif movement == 4: move_dir = (1, 0)  # Right

            if move_dir:
                current_node = self.network[self.player_grid_pos[1]][self.player_grid_pos[0]]
                if current_node[move_dir]:
                    self.player_grid_pos = (self.player_grid_pos[0] + move_dir[0], self.player_grid_pos[1] + move_dir[1])
                    self.player_target_pos = pygame.Vector2(self.player_grid_pos) * self.GRID_SIZE
                    self.is_moving = True
                    # sfx: player_move

        # --- Card Activation ---
        if space_pressed and self.card_cooldowns[0] == 0:
            self._activate_card(0)
        if shift_pressed and self.card_cooldowns[1] == 0:
            self._activate_card(1)

        self.prev_space_held, self.prev_shift_held = space_held, shift_held

    def _activate_card(self, hand_index):
        if hand_index >= len(self.player_hand): return
        card = self.player_hand[hand_index]
        if not card: return

        self.reward_this_step += 10 # Combo/Card use reward
        self.card_cooldowns[hand_index] = card["cooldown"]
        # sfx: card_activate
        
        if card["effect"] == "add_momentum":
            self.momentum = min(100, self.momentum + card["value"])
            self._create_particles(self.player_pos, 20, self.COLOR_MOMENTUM_BAR, 2.0, 4.0)
        elif card["effect"] == "shield":
            self.active_effects["shield"] = card["value"]
            self._create_particles(self.player_pos, 20, card["color"], 1.0, 3.0)
        elif card["effect"] == "pathfind":
            self.active_effects["pathfind"] = card["value"]
            self.shortest_path = self._find_shortest_path(self.player_grid_pos, self.exit_pos)
            self._create_particles(self.player_pos, 20, card["color"], 1.5, 3.5)

    def _update_player(self):
        # --- Interpolate Movement ---
        if self.is_moving:
            direction = self.player_target_pos - self.player_pos
            if direction.length() < self.PLAYER_SPEED:
                self.player_pos = self.player_target_pos
                self.is_moving = False
                self.momentum = min(100, self.momentum + 2.5) # Gain momentum on arrival
            else:
                self.player_pos += direction.normalize() * self.PLAYER_SPEED
                # Add trail particle
                self.trail_particles.append(self.player_pos.copy())
        
        # --- Update Momentum ---
        self.momentum -= 0.1 # Constant decay
        self.momentum = max(0, self.momentum)

        # --- Update Camera ---
        self.camera_pos.x += (self.player_pos.x - self.camera_pos.x) * 0.1
        self.camera_pos.y += (self.player_pos.y - self.camera_pos.y) * 0.1
    
    def _update_security_nodes(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 500 == 0:
            self.current_pulse_interval = max(0.5 * self.FPS, self.current_pulse_interval - (0.05 * self.FPS))

        for node in self.security_nodes:
            node["timer"] -= 1
            if node["timer"] <= 0:
                node["timer"] = self.current_pulse_interval
                node["pulse_state"] = 1.0 # Start pulse
                # sfx: security_pulse_charge
                
                # Check if player is caught
                dist_to_player = (pygame.Vector2(node["pos"]) * self.GRID_SIZE - self.player_pos).length()
                if dist_to_player < node["radius"] * self.GRID_SIZE:
                    if self.active_effects["shield"] > 0:
                        self.reward_this_step += 5 # Bypassed security
                        # sfx: shield_block
                    else:
                        self.game_over = True # Caught!

            # Animate pulse
            if node["pulse_state"] > 0:
                node["pulse_state"] -= 0.05
                if node["pulse_state"] < 0:
                    node["pulse_state"] = 0

    def _update_effects_and_cooldowns(self):
        for key in self.active_effects:
            if self.active_effects[key] > 0:
                self.active_effects[key] -= 1
                if key == "pathfind" and self.active_effects[key] == 0:
                    self.shortest_path = None # Pathfinder effect wears off

        for i in range(len(self.card_cooldowns)):
            if self.card_cooldowns[i] > 0:
                self.card_cooldowns[i] -= 1

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        if self.momentum > 20:
            self.reward_this_step += 0.01

    def _check_termination(self):
        return (
            self.game_over or
            self.momentum <= 0 or
            self.player_grid_pos == self.exit_pos or
            self.steps >= 5000
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
            "momentum": self.momentum,
            "unlocked_cards": len(self.unlocked_card_indices)
        }

    def _render_game(self):
        cam_offset = self.camera_pos - pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self._draw_network(cam_offset)
        if self.shortest_path and self.active_effects["pathfind"] > 0:
            self._draw_pathfinder_path(cam_offset)
        self._draw_nodes(cam_offset)
        self._draw_particles(cam_offset)
        self._draw_player(cam_offset)

    def _render_ui(self):
        # --- Score and Steps ---
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # --- Momentum Bar ---
        bar_width, bar_height = 200, 20
        bar_x, bar_y = self.WIDTH / 2 - bar_width / 2, 15
        fill_width = (self.momentum / 100) * bar_width
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # --- Cards ---
        if self.player_hand and len(self.player_hand) > 0:
            self._draw_card_ui(0, self.player_hand[0], self.card_cooldowns[0], (40, self.HEIGHT - 40))
        if self.player_hand and len(self.player_hand) > 1:
            self._draw_card_ui(1, self.player_hand[1], self.card_cooldowns[1], (self.WIDTH - 40, self.HEIGHT - 40))

        # --- Game Over Screen ---
        if self.game_over and self.player_grid_pos != self.exit_pos:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 100))
            self.screen.blit(overlay, (0,0))
            fail_text = self.font_large.render("CONNECTION SEVERED", True, (255, 200, 200))
            self.screen.blit(fail_text, (self.WIDTH/2 - fail_text.get_width()/2, self.HEIGHT/2 - fail_text.get_height()/2))

    def _draw_network(self, offset):
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                node = self.network[y][x]
                px, py = x * self.GRID_SIZE - offset.x, y * self.GRID_SIZE - offset.y
                if node[(0, 1)]: # South
                    pygame.draw.line(self.screen, self.COLOR_GRID, (px, py), (px, py + self.GRID_SIZE), 1)
                if node[(-1, 0)]: # West
                    pygame.draw.line(self.screen, self.COLOR_GRID, (px, py), (px - self.GRID_SIZE, py), 1)

    def _draw_nodes(self, offset):
        # Exit Node
        exit_px, exit_py = self.exit_pos[0] * self.GRID_SIZE - offset.x, self.exit_pos[1] * self.GRID_SIZE - offset.y
        self._draw_glow_circle(self.screen, self.COLOR_EXIT, (exit_px, exit_py), self.GRID_SIZE * 0.4, 15)

        # Security Nodes
        for node in self.security_nodes:
            node_px, node_py = node["pos"][0] * self.GRID_SIZE - offset.x, node["pos"][1] * self.GRID_SIZE - offset.y
            self._draw_glow_circle(self.screen, self.COLOR_SECURITY, (node_px, node_py), self.GRID_SIZE * 0.3, 10)
            if node["pulse_state"] > 0:
                pulse_rad = (1.0 - node["pulse_state"]) * node["radius"] * self.GRID_SIZE
                alpha = int(node["pulse_state"] * 150)
                color = (*self.COLOR_SECURITY, alpha)
                pygame.gfxdraw.aacircle(self.screen, int(node_px), int(node_py), int(pulse_rad), color)

    def _draw_player(self, offset):
        px, py = self.player_pos.x - offset.x, self.player_pos.y - offset.y
        
        # Shield effect
        if self.active_effects["shield"] > 0:
            shield_rad = self.GRID_SIZE * 0.5
            alpha = 100 + (self.active_effects["shield"] % 15) * 5 # Pulsing alpha
            self._draw_glow_circle(self.screen, self.AVAILABLE_CARDS[1]["color"], (px, py), shield_rad, 10, alpha)

        # Player core
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, (px, py), self.GRID_SIZE * 0.25, 20)

    def _draw_particles(self, offset):
        # Trail
        if len(self.trail_particles) > 1:
            for i in range(len(self.trail_particles) - 1):
                p1 = self.trail_particles[i] - offset
                p2 = self.trail_particles[i+1] - offset
                alpha = int((i / len(self.trail_particles)) * 255)
                color = (*self.COLOR_TRAIL[:3], alpha)
                pygame.draw.line(self.screen, color, p1, p2, max(1, int(i / len(self.trail_particles) * 4)))
        
        # General particles
        for p in self.particles:
            px, py = p['pos'] - offset
            alpha = int((p['life'] / p['max_life']) * 255)
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(p['size']), color)

    def _draw_pathfinder_path(self, offset):
        if not self.shortest_path or len(self.shortest_path) < 2: return
        
        alpha = 50 + (self.active_effects["pathfind"] % 20) * 5 # Pulsing alpha
        color = (*self.AVAILABLE_CARDS[2]["color"], alpha)

        points = [(pygame.Vector2(p) * self.GRID_SIZE - offset) for p in self.shortest_path]
        pygame.draw.aalines(self.screen, color, False, points, 3)

    def _draw_card_ui(self, index, card, cooldown, pos):
        radius = 30
        if not card: return
        
        # Cooldown pie
        if cooldown > 0:
            angle = (cooldown / card["cooldown"]) * 360
            rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius*2, radius*2)
            pygame.draw.arc(self.screen, (100,100,100), rect, math.radians(90), math.radians(90+angle), radius)
        
        # Card circle
        self._draw_glow_circle(self.screen, card["color"], pos, radius * 0.7, 10)
        key_text = self.font_small.render("SPACE" if index == 0 else "SHIFT", True, self.COLOR_UI_TEXT)
        self.screen.blit(key_text, (pos[0] - key_text.get_width()/2, pos[1] + radius - 10))

    def _draw_glow_circle(self, surface, color, pos, radius, intensity, max_alpha=255):
        for i in range(intensity):
            alpha = max(0, max_alpha - (i * (max_alpha/intensity)))
            new_radius = radius + i * 0.5
            c = (*color, int(alpha))
            pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(new_radius), c)

    def _generate_network(self):
        self.grid_w, self.grid_h = 40, 30
        self.network = [[{(0,1):False, (0,-1):False, (1,0):False, (-1,0):False} for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        
        stack = []
        visited = set()
        
        x, y = self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h)
        stack.append((x,y))
        visited.add((x,y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and (nx, ny) not in visited:
                    neighbors.append(((dx, dy), (nx, ny)))
            
            if neighbors:
                (dx, dy), (nx, ny) = self.np_random.choice(neighbors, 1)[0]
                self.network[cy][cx][(dx, dy)] = True
                self.network[ny][nx][(-dx, -dy)] = True
                visited.add((nx,ny))
                stack.append((nx,ny))
            else:
                stack.pop()
    
    def _place_exit_node(self, start_pos):
        # Find a node far from the start
        q = deque([(start_pos, 0)])
        visited = {start_pos}
        farthest_node, max_dist = start_pos, 0
        while q:
            (x,y), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_node = (x,y)
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                if self.network[y][x][(dx,dy)]:
                    neighbor = (x+dx, y+dy)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist+1))
        self.exit_pos = farthest_node
        assert self.exit_pos != start_pos

    def _place_security_nodes(self, start_pos, exit_pos):
        self.security_nodes = []
        num_nodes = 5 + self.steps // 1000 # More nodes as game progresses
        for _ in range(num_nodes):
            while True:
                pos = (self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h))
                if pos != start_pos and pos != exit_pos and pos not in [n['pos'] for n in self.security_nodes]:
                    self.security_nodes.append({
                        "pos": pos,
                        "timer": self.np_random.integers(0, int(self.current_pulse_interval)),
                        "radius": self.np_random.uniform(1.5, 2.5),
                        "pulse_state": 0.0
                    })
                    break
    
    def _find_shortest_path(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            (x,y), path = q.popleft()
            if (x,y) == end:
                return path
            
            node = self.network[y][x]
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                if node[(dx,dy)]:
                    neighbor = (x+dx, y+dy)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = list(path)
                        new_path.append(neighbor)
                        q.append((neighbor, new_path))
        return None

    def _draw_hand(self):
        self.card_cooldowns = [0, 0]
        unlocked = list(self.unlocked_card_indices)
        
        card1_idx = self.np_random.choice(unlocked)
        card1 = self.AVAILABLE_CARDS[card1_idx]
        
        remaining_unlocked = [i for i in unlocked if i != card1_idx]
        if remaining_unlocked:
            card2_idx = self.np_random.choice(remaining_unlocked)
            card2 = self.AVAILABLE_CARDS[card2_idx]
        else:
            card2 = None # Not enough cards for a second slot
            
        self.player_hand = [card1, card2]

    def _unlock_new_card(self):
        locked_cards = [i for i, _ in enumerate(self.AVAILABLE_CARDS) if i not in self.unlocked_card_indices]
        if locked_cards:
            new_unlock = self.np_random.choice(locked_cards)
            self.unlocked_card_indices.add(new_unlock)
            # sfx: new_unlock

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and will not be run by the evaluation system.
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cyber Grid Runner")
    clock = pygame.time.Clock()

    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()