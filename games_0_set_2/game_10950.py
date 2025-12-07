import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A puzzle platformer Gymnasium environment. The agent controls a character
    who must build structures out of bubbles to reach an exit. Bubbles can trap
    enemies, and their combined density affects their buoyancy, creating a
    physics-based puzzle.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle platformer where you build structures from bubbles to reach an exit, "
        "while trapping enemies and managing bubble physics."
    )
    user_guide = (
        "Use ←→ arrow keys to move, ↑ to jump. Press space to shoot a bubble. "
        "Press shift to cycle through bubble types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.PLAYER_SPEED = 5
        self.PLAYER_JUMP_STRENGTH = -8 # Negative is up
        self.PLAYER_GRAVITY = 0.4
        self.PLAYER_RADIUS = 12
        self.BUBBLE_MAX_COUNT = 50
        self.BUBBLE_LIFESPAN = 500
        self.BUBBLE_SHOOT_COOLDOWN = 10 # steps
        self.BUBBLE_RADIUS = 20
        self.ENEMY_SPAWN_INTERVAL = 200
        self.ENEMY_GRAVITY = 0.2

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PLAYER = (255, 215, 0) # Yellow
        self.COLOR_PLAYER_GLOW = (255, 215, 0, 60)
        self.COLOR_EXIT = (138, 43, 226) # Purple
        self.COLOR_TEXT = (230, 230, 240)

        # Bubble types: (name, density, color)
        self.BUBBLE_TYPES = [
            {"name": "Low Density", "density": -0.2, "color": (0, 191, 255)}, # Light Blue
            {"name": "Medium Density", "density": 0.0, "color": (50, 205, 50)}, # Green
            {"name": "High Density", "density": 0.2, "color": (255, 69, 0)}, # Red
        ]

        # Enemy types: (weight, color, vertices)
        self.ENEMY_TYPES = [
            {"weight": 0.1, "color": (221, 160, 221), "vertices": 3}, # Light Purple, Triangle
            {"weight": 0.25, "color": (186, 85, 211), "vertices": 4}, # Medium Purple, Square
            {"weight": 0.4, "color": (148, 0, 211), "vertices": 5}, # Dark Purple, Pentagon
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.large_font = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.bubbles = None
        self.enemies = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.exit_pos = None
        self.exit_radius = None
        self.current_bubble_type_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        self.bubble_shoot_timer = None
        self.enemy_spawn_timer = None
        self.active_enemy_types = None
        self.last_dist_to_exit = None
        self.previous_structures = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.bubbles = []
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.exit_pos = pygame.Vector2(self.WIDTH / 2, 40)
        self.exit_radius = 25
        self.current_bubble_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.bubble_shoot_timer = 0
        self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL // 2
        self.active_enemy_types = [0]
        self.last_dist_to_exit = self.player_pos.distance_to(self.exit_pos)
        self.previous_structures = set()

        self._spawn_enemy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Input and Actions ---
        shoot_action = space_held and not self.last_space_held and self.bubble_shoot_timer <= 0
        cycle_action = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if cycle_action:
            self.current_bubble_type_idx = (self.current_bubble_type_idx + 1) % len(self.BUBBLE_TYPES)
            # SFX: Cycle sound

        self._update_player(movement)

        # --- Update Game State ---
        self._update_timers()
        self._spawn_enemy()
        if shoot_action:
            self._shoot_bubble()

        trapped_enemy_reward = self._update_enemies()
        reward += trapped_enemy_reward

        self._update_bubbles()
        self._update_particles()

        structure_reward = self._handle_bubble_collisions_and_structures()
        reward += structure_reward

        # --- Calculate Rewards ---
        dist_to_exit = self.player_pos.distance_to(self.exit_pos)
        reward += (self.last_dist_to_exit - dist_to_exit) * 0.01
        self.last_dist_to_exit = dist_to_exit

        self.score += reward

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_pos.y > self.HEIGHT + self.PLAYER_RADIUS:
            reward -= 100
            self.score -= 100
            terminated = True
            # SFX: Player fall
        elif self.player_pos.distance_to(self.exit_pos) < self.PLAYER_RADIUS + self.exit_radius:
            reward += 100
            self.score += 100
            terminated = True
            # SFX: Victory
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_timers(self):
        self.steps += 1
        if self.bubble_shoot_timer > 0:
            self.bubble_shoot_timer -= 1
        if self.enemy_spawn_timer > 0:
            self.enemy_spawn_timer -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            next_type = len(self.active_enemy_types)
            if next_type < len(self.ENEMY_TYPES):
                self.active_enemy_types.append(next_type)

    def _update_player(self, movement):
        # Movement
        if movement == 1: # Up
            on_bubble = False
            for b in self.bubbles:
                if self.player_pos.distance_to(b['pos']) < self.PLAYER_RADIUS + b['radius'] and self.player_pos.y < b['pos'].y:
                    on_bubble = True
                    break
            on_ground = self.player_pos.y >= self.HEIGHT - self.PLAYER_RADIUS
            if on_bubble or on_ground:
                self.player_vel.y = self.PLAYER_JUMP_STRENGTH
                # SFX: Jump
        if movement == 2: # Down
             self.player_vel.y += self.PLAYER_SPEED * 0.1
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        if movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED

        if movement == 0: # No horizontal movement
            self.player_vel.x *= 0.8 # Friction

        # Physics
        self.player_vel.y += self.PLAYER_GRAVITY
        self.player_pos += self.player_vel

        # Player-bubble collision
        for b in self.bubbles:
            dist = self.player_pos.distance_to(b['pos'])
            overlap = self.PLAYER_RADIUS + b['radius'] - dist
            if overlap > 0:
                if self.player_pos.y < b['pos'].y and self.player_vel.y > 0:
                    self.player_pos.y = b['pos'].y - self.PLAYER_RADIUS - b['radius']
                    self.player_vel.y = b['vel'].y
                    self.player_vel.x *= 0.95
                else:
                    if dist > 1e-6:
                        push_vec = (self.player_pos - b['pos']).normalize() * overlap
                        self.player_pos += push_vec
                        self.player_vel += push_vec * 0.1

        # Boundaries
        self.player_pos.x = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, self.player_pos.x))
        # Floor boundary
        if self.player_pos.y > self.HEIGHT - self.PLAYER_RADIUS:
            self.player_pos.y = self.HEIGHT - self.PLAYER_RADIUS
            if self.player_vel.y > 0:
                self.player_vel.y = 0

    def _spawn_enemy(self):
        if self.enemy_spawn_timer <= 0:
            enemy_type_idx = self.np_random.choice(self.active_enemy_types)
            enemy_type = self.ENEMY_TYPES[enemy_type_idx]
            self.enemies.append({
                "pos": pygame.Vector2(self.np_random.integers(50, self.WIDTH - 50), -20),
                "vel": pygame.Vector2(0, 0),
                "radius": 10,
                "type_idx": enemy_type_idx,
                "weight": enemy_type["weight"],
            })
            self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL

    def _shoot_bubble(self):
        if len(self.bubbles) < self.BUBBLE_MAX_COUNT:
            bubble_type = self.BUBBLE_TYPES[self.current_bubble_type_idx]
            self.bubbles.append({
                "pos": pygame.Vector2(self.player_pos),
                "vel": pygame.Vector2(self.player_vel.x * 0.5, -2),
                "radius": self.BUBBLE_RADIUS,
                "type_idx": self.current_bubble_type_idx,
                "density": bubble_type["density"],
                "color": bubble_type["color"],
                "trapped_enemy": None,
                "life": self.BUBBLE_LIFESPAN,
                "id": self.steps + self.np_random.random()
            })
            self.bubble_shoot_timer = self.BUBBLE_SHOOT_COOLDOWN
            # SFX: Shoot bubble

    def _update_enemies(self):
        trapped_reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            enemy['vel'].y += self.ENEMY_GRAVITY
            enemy['pos'] += enemy['vel']

            for bubble in self.bubbles:
                if bubble['trapped_enemy'] is None and enemy['pos'].distance_to(bubble['pos']) < bubble['radius']:
                    bubble['trapped_enemy'] = enemy
                    enemies_to_remove.append(i)
                    trapped_reward += 1
                    # SFX: Enemy trapped
                    break

            if enemy['pos'].y > self.HEIGHT + 50:
                 enemies_to_remove.append(i)

        for i in sorted(list(set(enemies_to_remove)), reverse=True):
            del self.enemies[i]

        return trapped_reward

    def _update_bubbles(self):
        bubbles_to_remove = []
        for i, b in enumerate(self.bubbles):
            b['life'] -= 1
            if b['life'] <= 0:
                bubbles_to_remove.append(i)
                self._create_pop_particles(b['pos'], b['color'])
                # SFX: Bubble pop
                if b['trapped_enemy']:
                    b['trapped_enemy']['pos'] = pygame.Vector2(b['pos'])
                    self.enemies.append(b['trapped_enemy'])
                continue

            force = pygame.Vector2(0, 0)
            enemy_weight = b['trapped_enemy']['weight'] if b['trapped_enemy'] else 0
            buoyancy = b['density'] - enemy_weight
            force.y += buoyancy

            b['vel'] += force
            b['vel'] *= 0.95 # Drag
            b['pos'] += b['vel']

            b['pos'].x = max(b['radius'], min(self.WIDTH - b['radius'], b['pos'].x))
            b['pos'].y = max(b['radius'], min(self.HEIGHT - b['radius'], b['pos'].y))

        for i in sorted(bubbles_to_remove, reverse=True):
            del self.bubbles[i]

    def _handle_bubble_collisions_and_structures(self):
        for i in range(len(self.bubbles)):
            for j in range(i + 1, len(self.bubbles)):
                b1, b2 = self.bubbles[i], self.bubbles[j]
                dist_vec = b1['pos'] - b2['pos']
                dist = dist_vec.length()
                min_dist = b1['radius'] + b2['radius']
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    push_vec = dist_vec.normalize() * overlap
                    b1['pos'] += push_vec * 0.5
                    b2['pos'] -= push_vec * 0.5

        reward = 0
        if not self.bubbles:
            self.previous_structures = set()
            return 0

        adj = {i: [] for i in range(len(self.bubbles))}
        for i in range(len(self.bubbles)):
            for j in range(i + 1, len(self.bubbles)):
                if self.bubbles[i]['pos'].distance_to(self.bubbles[j]['pos']) < self.bubbles[i]['radius'] + self.bubbles[j]['radius'] + 2:
                    adj[i].append(j)
                    adj[j].append(i)

        visited, current_structures = set(), set()
        for i in range(len(self.bubbles)):
            if i not in visited:
                component, q = set(), [i]
                visited.add(i)
                head = 0
                while head < len(q):
                    u = q[head]; head += 1
                    component.add(u)
                    for v in adj[u]:
                        if v not in visited: visited.add(v); q.append(v)
                current_structures.add(frozenset(component))

        for s in current_structures:
            if len(s) >= 3 and s not in self.previous_structures:
                reward += 5
                # SFX: Structure formed

        self.previous_structures = current_structures
        return reward

    def _create_pop_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": 20, "color": color, "radius": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1; p['pos'] += p['vel']
            p['vel'] *= 0.9; p['radius'] *= 0.95

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        wobble = math.sin(self.steps * 0.05) * 5
        for i in range(5):
            alpha = 150 - i * 30
            radius = self.exit_radius + wobble + i * 3
            pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), int(radius), self.COLOR_EXIT + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), int(radius), self.COLOR_EXIT + (alpha,))

        for enemy in self.enemies:
            e_type = self.ENEMY_TYPES[enemy['type_idx']]
            self._draw_polygon(self.screen, e_type['color'], e_type['vertices'], enemy['pos'], enemy['radius'])

        for b in self.bubbles:
            pos_x, pos_y = int(b['pos'].x), int(b['pos'].y)
            wobble = math.sin(self.steps * 0.2 + pos_x) * 1.5
            radius = int(b['radius'] + wobble)
            alpha = max(50, int(200 * (b['life'] / self.BUBBLE_LIFESPAN)))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, b['color'] + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, b['color'] + (alpha,))
            pygame.gfxdraw.filled_circle(self.screen, pos_x + radius // 3, pos_y - radius // 3, radius // 4, (255, 255, 255, alpha//2))
            if b['trapped_enemy']:
                e_type = self.ENEMY_TYPES[b['trapped_enemy']['type_idx']]
                self._draw_polygon(self.screen, e_type['color'], e_type['vertices'], b['pos'], b['trapped_enemy']['radius'])

        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'] + (alpha,))

        px, py = int(self.player_pos.x), int(self.player_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)

        indicator_pos = self.player_pos + pygame.Vector2(25, -25)
        bubble_type = self.BUBBLE_TYPES[self.current_bubble_type_idx]
        pygame.gfxdraw.filled_circle(self.screen, int(indicator_pos.x), int(indicator_pos.y), 8, bubble_type['color'] + (180,))
        pygame.gfxdraw.aacircle(self.screen, int(indicator_pos.x), int(indicator_pos.y), 8, bubble_type['color'])

    def _draw_polygon(self, surface, color, num_vertices, position, radius):
        angle_step, points = 2 * math.pi / num_vertices, []
        for i in range(num_vertices):
            angle = i * angle_step + (self.steps * 0.02)
            points.append((int(position.x + radius * math.cos(angle)), int(position.y + radius * math.sin(angle))))
        if len(points) >= 3:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "VICTORY!" if self.player_pos.distance_to(self.exit_pos) < self.PLAYER_RADIUS + self.exit_radius else "GAME OVER"
            end_text = self.large_font.render(msg, True, self.COLOR_PLAYER)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    pygame.display.set_caption("Bubble Structure Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done, running = False, True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()