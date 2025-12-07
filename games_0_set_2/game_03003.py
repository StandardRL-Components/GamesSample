
# Generated: 2025-08-28T06:41:11.309378
# Source Brief: brief_03003.md
# Brief Index: 3003

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Shift to rotate attack direction. Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot in a grid-based arena. Destroy all enemies before they destroy you."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 1000
        self.INITIAL_ENEMIES = 20
        self.INITIAL_ROBOT_HEALTH = 5

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 120, 120)
        self.COLOR_LASER = (255, 255, 100)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 150, 0), (200, 50, 0)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (40, 200, 40)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_large = pygame.font.SysFont('monospace', 64, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.robot_pos = pygame.Vector2(0, 0)
        self.robot_health = 0
        self.robot_facing_dir = 0 # 0:R, 1:U, 2:L, 3:D
        self.enemies = []
        self.enemy_count = 0
        self.particles = []
        self.attack_visuals = []
        self.damage_flash_timer = 0

        self.reset()

    def _get_direction_vector(self, direction_int):
        if direction_int == 0: return pygame.Vector2(1, 0) # Right
        if direction_int == 1: return pygame.Vector2(0, -1) # Up
        if direction_int == 2: return pygame.Vector2(-1, 0) # Left
        if direction_int == 3: return pygame.Vector2(0, 1) # Down
        return pygame.Vector2(0, 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.robot_health = self.INITIAL_ROBOT_HEALTH
        self.robot_facing_dir = 0  # Start facing right
        self.enemy_count = self.INITIAL_ENEMIES
        self.particles = []
        self.attack_visuals = []
        self.damage_flash_timer = 0
        
        # --- Place entities ---
        all_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_positions)
        
        player_pos_tuple = all_positions.pop()
        self.robot_pos = pygame.Vector2(player_pos_tuple[0], player_pos_tuple[1])
        
        self.enemies = []
        enemy_positions = all_positions[:self.INITIAL_ENEMIES]
        for pos in enemy_positions:
            self.enemies.append({
                "pos": pygame.Vector2(pos[0], pos[1]),
                "alive": True
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage efficiency
        self.attack_visuals.clear()
        
        # --- Handle Player Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Rotation
        if shift_held:
            self.robot_facing_dir = (self.robot_facing_dir + 1) % 4
            # sfx: rotate_sound

        # 2. Movement
        if movement != 0:
            move_dir_map = {1: 1, 2: 3, 3: 2, 4: 0} # map arrow keys to facing dir
            move_vec = self._get_direction_vector(move_dir_map[movement])
            new_pos = self.robot_pos + move_vec
            
            living_enemies = [e for e in self.enemies if e["alive"]]
            dist_before = float('inf')
            if living_enemies:
                dist_before = min([self.robot_pos.distance_to(e['pos']) for e in living_enemies])
            
            is_valid_move = (0 <= new_pos.x < self.GRID_WIDTH and
                             0 <= new_pos.y < self.GRID_HEIGHT and
                             all(new_pos != e['pos'] for e in living_enemies))

            if is_valid_move:
                self.robot_pos = new_pos
                if living_enemies:
                    dist_after = min([self.robot_pos.distance_to(e['pos']) for e in living_enemies])
                    if dist_after < dist_before:
                        reward += 0.5
                        
        # 3. Attack
        if space_held:
            # sfx: laser_fire
            attack_dir = self._get_direction_vector(self.robot_facing_dir)
            targets = []
            for enemy in self.enemies:
                if not enemy["alive"]: continue
                delta = enemy["pos"] - self.robot_pos
                if attack_dir.x != 0 and delta.y == 0 and np.sign(delta.x) == attack_dir.x:
                    targets.append((enemy, delta.magnitude()))
                elif attack_dir.y != 0 and delta.x == 0 and np.sign(delta.y) == attack_dir.y:
                    targets.append((enemy, delta.magnitude()))

            if targets:
                closest_enemy, _ = min(targets, key=lambda t: t[1])
                closest_enemy["alive"] = False
                self.enemy_count -= 1
                reward += 10
                self._create_explosion(closest_enemy["pos"])
                # sfx: explosion
                self.attack_visuals.append((self.robot_pos, closest_enemy["pos"]))
            else:
                 end_point = self.robot_pos + attack_dir * max(self.GRID_WIDTH, self.GRID_HEIGHT)
                 self.attack_visuals.append((self.robot_pos, end_point))

        # --- Handle Enemy Turn ---
        enemy_positions = {tuple(e['pos']) for e in self.enemies if e['alive']}
        enemy_positions.add(tuple(self.robot_pos))

        for enemy in self.enemies:
            if not enemy["alive"]: continue
            if enemy["pos"].distance_to(self.robot_pos) == 1:
                self.robot_health -= 1
                reward -= 1
                self.damage_flash_timer = 3
                # sfx: player_hit
                continue
            dx, dy = self.robot_pos.x - enemy["pos"].x, self.robot_pos.y - enemy["pos"].y
            possible_moves = []
            if dx != 0: possible_moves.append(pygame.Vector2(np.sign(dx), 0))
            if dy != 0: possible_moves.append(pygame.Vector2(0, np.sign(dy)))
            if not possible_moves: continue
            self.np_random.shuffle(possible_moves)
            for move in possible_moves:
                new_pos = enemy["pos"] + move
                if tuple(new_pos) not in enemy_positions:
                    enemy_positions.remove(tuple(enemy['pos']))
                    enemy['pos'] = new_pos
                    enemy_positions.add(tuple(new_pos))
                    break
        
        self.steps += 1
        self._update_particles()
        if self.damage_flash_timer > 0: self.damage_flash_timer -= 1
        
        self.score += reward

        # --- Check Termination ---
        terminated = False
        if self.robot_health <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100
        elif self.enemy_count == 0:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos * self.CELL_SIZE + pygame.Vector2(self.CELL_SIZE/2, self.CELL_SIZE/2),
                "vel": vel,
                "lifetime": self.np_random.integers(10, 25),
                "color": self.COLOR_EXPLOSION[self.np_random.integers(len(self.COLOR_EXPLOSION))]
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['vel'] *= 0.95

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self.damage_flash_timer > 0:
            self.screen.fill((100,0,0), special_flags=pygame.BLEND_RGB_ADD)

        self._render_grid()
        self._render_enemies()
        self._render_player()
        self._render_attack_visuals()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_player(self):
        center_px = self.robot_pos * self.CELL_SIZE + pygame.Vector2(self.CELL_SIZE/2, self.CELL_SIZE/2)
        radius = int(self.CELL_SIZE * 0.4)
        
        pygame.gfxdraw.filled_circle(self.screen, int(center_px.x), int(center_px.y), radius + 3, self.COLOR_PLAYER_GLOW + (50,))
        pygame.gfxdraw.aacircle(self.screen, int(center_px.x), int(center_px.y), radius + 3, self.COLOR_PLAYER_GLOW + (80,))
        pygame.gfxdraw.filled_circle(self.screen, int(center_px.x), int(center_px.y), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(center_px.x), int(center_px.y), radius, self.COLOR_PLAYER)

        dir_vec = self._get_direction_vector(self.robot_facing_dir)
        p1 = center_px + dir_vec * radius
        p2 = center_px + dir_vec.rotate(90) * (radius * 0.5)
        p3 = center_px + dir_vec.rotate(-90) * (radius * 0.5)
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER_GLOW, [p1, p2, p3])

    def _render_enemies(self):
        size = int(self.CELL_SIZE * 0.8)
        offset = (self.CELL_SIZE - size) / 2
        for enemy in self.enemies:
            if enemy["alive"]:
                rect = pygame.Rect(enemy["pos"].x * self.CELL_SIZE + offset, enemy["pos"].y * self.CELL_SIZE + offset, size, size)
                glow_rect = rect.inflate(4,4)
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_GLOW + (50,), glow_rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)

    def _render_attack_visuals(self):
        for start_grid, end_grid in self.attack_visuals:
            start_px = start_grid * self.CELL_SIZE + pygame.Vector2(self.CELL_SIZE/2, self.CELL_SIZE/2)
            end_px = end_grid * self.CELL_SIZE + pygame.Vector2(self.CELL_SIZE/2, self.CELL_SIZE/2)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_px, end_px, 4)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 25.0))
            color_with_alpha = p['color'] + (int(alpha),)
            size = max(1, int(p['lifetime'] * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), size, color_with_alpha)

    def _render_ui(self):
        health_ratio = max(0, self.robot_health / self.INITIAL_ROBOT_HEALTH)
        bar_width, bar_height = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))
        
        text_surf = self.font_small.render(f"Enemies: {self.enemy_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text = "VICTORY" if self.win else "YOU DIED"
        color = (100, 255, 100) if self.win else (255, 50, 50)
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return { "score": self.score, "steps": self.steps }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Grid Annihilator")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    running = True
    action_to_perform = [0, 0, 0] # Start with no-op

    while running:
        # --- Event handling to determine next action ---
        movement, space_held, shift_held = 0, 0, 0
        
        # This logic ensures a single key press corresponds to a single turn
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    done = False
                
                # Set action based on key press
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

                # If any action key was pressed, perform a step
                action_to_perform = [movement, space_held, shift_held]
                if not done:
                    if any(action_to_perform):
                        obs, reward, terminated, truncated, info = env.step(action_to_perform)
                        done = terminated or truncated
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)

    env.close()