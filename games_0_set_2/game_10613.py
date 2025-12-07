import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:41:34.265026
# Source Brief: brief_00613.md
# Brief Index: 613
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent, a glowing root tip, navigates a
    procedurally generated tunnel system. The goal is to collect sequences of
    nutrient nodes to power up an attack, and then launch projectiles at the
    enemy plant's heart. The agent can also flip gravity to navigate the tunnels.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a glowing root tip through a tunnel system to attack an enemy heart. "
        "Collect nutrient nodes in sequence to power up your attack and flip gravity to traverse the tunnels."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your root tip. Press 'space' to fire a projectile. "
        "Press 'shift' to flip gravity."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 10, 5)
    COLOR_TUNNEL = (50, 40, 30)
    COLOR_TUNNEL_BORDER = (40, 30, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 40)
    COLOR_HEART = (255, 20, 20)
    COLOR_HEART_GLOW = (255, 20, 20, 60)
    COLOR_ATTACK = (255, 150, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BAR_BG = (70, 70, 70)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_POWER_BAR = (255, 200, 0)
    NODE_COLORS = {
        "blue": (100, 100, 255),
        "yellow": (255, 255, 100),
        "purple": (200, 100, 255)
    }

    # Game Parameters
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 8
    HEART_RADIUS = 20
    NODE_RADIUS = 6
    GRID_SIZE = 20
    TUNNEL_WALK_STEPS = 250
    NUM_NODES = 15
    TARGET_SEQUENCE_LENGTH = 3
    MAX_ATTACK_POWER = 100
    ATTACK_SPEED = 12.0
    ATTACK_DAMAGE_FACTOR = 0.5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_health = 0
        self.gravity_up = False
        self.aim_direction = pygame.math.Vector2(1, 0)
        
        self.enemy_heart_pos = pygame.math.Vector2(0, 0)
        self.enemy_heart_health = 0
        self.initial_heart_health = 100
        
        self.attack_power = 0
        self.projectiles = []
        self.particles = []
        
        self.tunnel_tiles = set()
        self.nodes = []
        self.target_sequence = []
        self.collected_sequence = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # For edge-triggered actions
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_level()

        self.player_health = 100.0
        self.gravity_up = False
        self.aim_direction = pygame.math.Vector2(1, 0)
        
        self.enemy_heart_health = self.initial_heart_health
        
        self.attack_power = 0
        self.projectiles.clear()
        self.particles.clear()
        
        self.collected_sequence.clear()
        self._generate_target_sequence()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small cost for existing

        # Handle edge-triggered actions
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # Update game logic
        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_player(movement)
        
        # Update projectiles and check for heart hits
        damage_reward = self._update_projectiles()
        reward += damage_reward

        self._update_particles()
        
        # Check for node collection
        collection_reward = self._check_node_collisions()
        reward += collection_reward

        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if self.enemy_heart_health <= 0:
                reward += 100.0  # Victory bonus
                # Increase difficulty for next round
                self.initial_heart_health += 20
            elif self.player_health <= 0:
                reward -= 100.0  # Defeat penalty
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        self.tunnel_tiles.clear()
        self.nodes.clear()
        
        grid_w = self.SCREEN_WIDTH // self.GRID_SIZE
        grid_h = self.SCREEN_HEIGHT // self.GRID_SIZE
        
        start_x, start_y = self.np_random.integers(1, grid_w-1), self.np_random.integers(1, grid_h-1)
        cx, cy = start_x, start_y
        
        self.tunnel_tiles.add((cx, cy))
        
        for _ in range(self.TUNNEL_WALK_STEPS):
            dx, dy = self.np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            cx = np.clip(cx + dx, 1, grid_w - 2)
            cy = np.clip(cy + dy, 1, grid_h - 2)
            self.tunnel_tiles.add((cx, cy))
        
        tunnel_list = list(self.tunnel_tiles)
        player_grid_pos = tunnel_list[0]
        self.player_pos = self._grid_to_world(player_grid_pos)

        heart_grid_pos = tunnel_list[-1]
        self.enemy_heart_pos = self._grid_to_world(heart_grid_pos)
        
        # Place nodes
        node_positions = self.np_random.choice(len(tunnel_list), size=self.NUM_NODES, replace=False)
        for i in node_positions:
            node_grid_pos = tunnel_list[i]
            if node_grid_pos != player_grid_pos and node_grid_pos != heart_grid_pos:
                node_type = self.np_random.choice(list(self.NODE_COLORS.keys()))
                self.nodes.append({
                    "pos": self._grid_to_world(node_grid_pos),
                    "type": node_type,
                    "color": self.NODE_COLORS[node_type]
                })
    
    def _generate_target_sequence(self):
        self.target_sequence = [self.np_random.choice(list(self.NODE_COLORS.keys())) for _ in range(self.TARGET_SEQUENCE_LENGTH)]

    def _handle_input(self, movement, space_pressed, shift_pressed):
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right

        if move_vec.length_squared() > 0:
            self.aim_direction = move_vec.normalize()

        if shift_pressed:
            self.gravity_up = not self.gravity_up
            # SFX: Gravity shift sound

        if space_pressed and self.attack_power > 0:
            self._launch_attack()
            # SFX: Attack launch sound

    def _update_player(self, movement):
        move_vec = pygame.math.Vector2(0, 0)
        
        # Apply gravity to vertical movement
        vertical_mult = -1 if self.gravity_up else 1
        if movement == 1: move_vec.y = -1 * vertical_mult # Up
        elif movement == 2: move_vec.y = 1 * vertical_mult # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right

        if move_vec.length_squared() > 0:
            new_pos = self.player_pos + move_vec.normalize() * self.PLAYER_SPEED
            if self._is_in_tunnel(new_pos):
                self.player_pos = new_pos
        
        # Player takes damage from being outside tunnels
        if not self._is_in_tunnel(self.player_pos):
            self.player_health -= 0.5
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 1, 0.2)


    def _launch_attack(self):
        damage = self.attack_power * self.ATTACK_DAMAGE_FACTOR
        self.projectiles.append({
            "pos": self.player_pos.copy(),
            "vel": self.aim_direction * self.ATTACK_SPEED,
            "damage": damage,
        })
        self.attack_power = 0
    
    def _update_projectiles(self):
        damage_reward = 0
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            
            # Check collision with heart
            if p["pos"].distance_to(self.enemy_heart_pos) < self.HEART_RADIUS:
                self.enemy_heart_health -= p["damage"]
                self.enemy_heart_health = max(0, self.enemy_heart_health)
                damage_reward += 10.0
                self._create_particles(p["pos"], self.COLOR_HEART, 20, 3.0)
                self.projectiles.remove(p)
                # SFX: Heart impact sound
                continue

            # Check collision with walls
            if not self._is_in_tunnel(p["pos"], margin=5):
                self._create_particles(p["pos"], self.COLOR_ATTACK, 5, 1.5)
                self.projectiles.remove(p)
                # SFX: Projectile fizzle sound
                continue
        return damage_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_node_collisions(self):
        reward = 0
        for node in self.nodes[:]:
            if self.player_pos.distance_to(node["pos"]) < self.PLAYER_RADIUS + self.NODE_RADIUS:
                reward += self._collect_node(node)
                self.nodes.remove(node)
                # SFX: Node collection sound
                self._create_particles(node["pos"], node["color"], 10, 2.0)
        return reward
    
    def _collect_node(self, node):
        self.collected_sequence.append(node["type"])
        
        # Check if collected sequence matches target
        is_correct_sequence = (self.collected_sequence == self.target_sequence[:len(self.collected_sequence)])
        
        if is_correct_sequence:
            power_gain = 15
            if len(self.collected_sequence) == len(self.target_sequence):
                # Sequence complete bonus
                power_gain += 30
                self.collected_sequence.clear()
                self._generate_target_sequence()
        else:
            # Wrong node, sequence breaks
            power_gain = 5
            self.collected_sequence.clear()
        
        self.attack_power = min(self.MAX_ATTACK_POWER, self.attack_power + power_gain)
        return 1.0 # Base reward for any collection

    def _check_termination(self):
        return (self.player_health <= 0 or 
                self.enemy_heart_health <= 0)

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
            "player_health": self.player_health,
            "enemy_health": self.enemy_heart_health,
            "attack_power": self.attack_power,
        }
        
    def _render_game(self):
        # Tunnels
        for gx, gy in self.tunnel_tiles:
            rect = pygame.Rect(gx * self.GRID_SIZE, gy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TUNNEL_BORDER, rect.inflate(4, 4))
            pygame.draw.rect(self.screen, self.COLOR_TUNNEL, rect)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

        # Nodes
        for node in self.nodes:
            pos = (int(node["pos"].x), int(node["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.NODE_RADIUS, node["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.NODE_RADIUS, node["color"])

        # Aim indicator
        if self.attack_power > 0:
            start_pos = self.player_pos + self.aim_direction * (self.PLAYER_RADIUS + 5)
            for i in range(8):
                p1 = start_pos + self.aim_direction * i * 10
                p2 = p1 + self.aim_direction * 5
                pygame.draw.line(self.screen, self.COLOR_ATTACK, p1, p2, 1)

        # Enemy Heart
        pulse = 1 + 0.1 * math.sin(self.steps * 0.1)
        heart_r = int(self.HEART_RADIUS * pulse)
        heart_pos = (int(self.enemy_heart_pos.x), int(self.enemy_heart_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, heart_pos[0], heart_pos[1], heart_r + 5, self.COLOR_HEART_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, heart_pos[0], heart_pos[1], heart_r, self.COLOR_HEART)
        pygame.gfxdraw.aacircle(self.screen, heart_pos[0], heart_pos[1], heart_r, self.COLOR_HEART)

        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS + 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.draw.circle(self.screen, self.COLOR_ATTACK, pos, 4)

    def _render_ui(self):
        # Health Bar
        self._draw_bar(10, 10, 150, 20, self.player_health / 100.0, self.COLOR_HEALTH_BAR, "HEALTH")
        # Attack Power Bar
        self._draw_bar(self.SCREEN_WIDTH - 160, 10, 150, 20, self.attack_power / self.MAX_ATTACK_POWER, self.COLOR_POWER_BAR, "POWER")

        # Gravity Indicator
        arrow_points = []
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25
        if self.gravity_up:
            arrow_points = [(cx, cy - 10), (cx - 10, cy + 5), (cx + 10, cy + 5)]
        else:
            arrow_points = [(cx, cy + 10), (cx - 10, cy - 5), (cx + 10, cy - 5)]
        pygame.draw.polygon(self.screen, self.COLOR_TEXT, arrow_points)
        pygame.gfxdraw.aapolygon(self.screen, arrow_points, self.COLOR_TEXT)

        # Target Sequence
        seq_text = self.font_small.render("TARGET:", True, self.COLOR_TEXT)
        self.screen.blit(seq_text, (10, self.SCREEN_HEIGHT - 30))
        for i, color_name in enumerate(self.target_sequence):
            color = self.NODE_COLORS[color_name]
            pygame.draw.rect(self.screen, color, (80 + i * 25, self.SCREEN_HEIGHT - 28, 20, 15))
            if i < len(self.collected_sequence) and self.collected_sequence[i] == color_name:
                 pygame.draw.rect(self.screen, self.COLOR_TEXT, (80 + i * 25, self.SCREEN_HEIGHT - 28, 20, 15), 2)


    def _draw_bar(self, x, y, w, h, percent, color, label):
        percent = max(0, min(1, percent))
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * percent), h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (x, y, w, h), 1)
        text = self.font_small.render(label, True, self.COLOR_TEXT)
        self.screen.blit(text, (x + (w - text.get_width()) // 2, y + (h - text.get_height()) // 2))

    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": life, "max_life": life,
                "color": color, "radius": self.np_random.integers(1, 4)
            })

    def _world_to_grid(self, pos):
        return (int(pos.x // self.GRID_SIZE), int(pos.y // self.GRID_SIZE))

    def _grid_to_world(self, grid_pos):
        return pygame.math.Vector2(
            grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2,
            grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2
        )

    def _is_in_tunnel(self, pos, margin=0):
        gx, gy = self._world_to_grid(pos)
        
        # Check a 3x3 area around the grid cell for more lenient collision
        for x in range(gx - 1, gx + 2):
            for y in range(gy - 1, gy + 2):
                if (x, y) in self.tunnel_tiles:
                    rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                    if rect.inflate(margin, margin).collidepoint(pos.x, pos.y):
                        return True
        return False
        
    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Root System Navigator")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False
                print("--- Game Reset ---")

        clock.tick(GameEnv.GAME_FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()