import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to attack in your last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated dungeon, battling enemies to reach the final room and claim its treasure. A turn-based RPG."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000
        self.NUM_ROOMS = 5

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 60)
        self.COLOR_FLOOR = (35, 35, 45)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_PLAYER_ACCENT = (150, 255, 150)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_ACCENT = (255, 150, 150)
        self.COLOR_TREASURE = (255, 200, 0)
        self.COLOR_EXIT = (150, 100, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR_FILL = (0, 180, 0)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_floating = pygame.font.Font(None, 20)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_max_health = None
        self.player_facing_dir = None
        self.player_damage_flash = 0
        self.enemies = []
        self.treasures = []
        self.particles = []
        self.floating_texts = []
        self.room_number = None
        self.score = None
        self.steps = None
        self.game_over = None

        # self.reset() # reset is called by the test harness

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.room_number = 1

        self.player_max_health = 100
        self.player_health = self.player_max_health
        self.player_facing_dir = np.array([0, -1])  # Start facing up
        self.player_damage_flash = 0

        self.particles = []
        self.floating_texts = []

        self._generate_room()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean

        reward = -0.1  # Cost of living
        terminated = False

        # --- Player Turn ---
        # 1. Handle Movement
        move_vec = self._get_move_vector(movement)

        # Calculate distance to nearest target before moving
        dist_before = self._get_dist_to_nearest_target()

        if np.any(move_vec):
            self.player_facing_dir = move_vec
            new_pos = self.player_pos + move_vec
            if self._is_walkable(new_pos):
                self.player_pos = new_pos

        # Reward for moving towards a target
        dist_after = self._get_dist_to_nearest_target()
        if np.any(move_vec):
            if dist_after is None or dist_before is None or dist_after >= dist_before:
                reward -= 2  # Penalty for "safe" or non-optimal move

        # 2. Check for Treasure Collection
        for treasure in self.treasures:
            if np.array_equal(self.player_pos, treasure['pos']) and not treasure['collected']:
                treasure['collected'] = True
                self.score += 2
                reward += 2
                self._add_floating_text("+2", self._grid_to_pixel(self.player_pos), self.COLOR_TREASURE)

        # 3. Handle Attack
        if space_held:
            reward += self._player_attack()

        # --- Enemy Turn ---
        for i, enemy in enumerate(self.enemies):
            action_taken = False
            # Check for adjacency to player
            if np.linalg.norm(self.player_pos - enemy['pos']) < 1.5:
                # Attack player
                self.player_health -= enemy['damage']
                self.player_damage_flash = 5  # frames
                reward -= 5  # Penalty for taking damage
                self._add_particles(self._grid_to_pixel(self.player_pos), self.COLOR_ENEMY, 10, 'impact')
                action_taken = True

            if not action_taken:
                # Patrol
                target_patrol_point = enemy['patrol_points'][enemy['patrol_index']]
                if np.array_equal(enemy['pos'], target_patrol_point):
                    enemy['patrol_index'] = (enemy['patrol_index'] + 1) % len(enemy['patrol_points'])
                    target_patrol_point = enemy['patrol_points'][enemy['patrol_index']]

                move_dir = np.clip(target_patrol_point - enemy['pos'], -1, 1).astype(int)
                new_enemy_pos = enemy['pos'] + move_dir

                # Check for collisions with other enemies or player
                can_move = True
                if np.array_equal(new_enemy_pos, self.player_pos):
                    can_move = False
                for j, other_enemy in enumerate(self.enemies):
                    if i != j and np.array_equal(new_enemy_pos, other_enemy['pos']):
                        can_move = False
                        break

                if can_move and self._is_walkable(new_enemy_pos):
                    enemy['pos'] = new_enemy_pos

        # --- Update State ---
        self.steps += 1
        self._update_effects()

        # --- Check for Room Transition ---
        if not self.enemies and np.array_equal(self.player_pos, self._get_exit_pos()):
            self.room_number += 1
            if self.room_number > self.NUM_ROOMS:
                # Victory!
                self.score += 100
                reward += 100
                terminated = True
                self.game_over = True
                self._add_floating_text("VICTORY!", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TREASURE, 60, 48)
            else:
                self._generate_room()
                reward += 20  # Reward for clearing a room

        # --- Check Termination Conditions ---
        if self.player_health <= 0:
            self.score -= 100
            reward -= 100
            terminated = True
            self.game_over = True
            self.player_health = 0
            self._add_floating_text("GAME OVER", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_ENEMY, 60, 48)

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_room(self):
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.enemies = []
        self.treasures = []

        occupied_positions = {tuple(self.player_pos)}

        # Place enemies
        enemy_health = 10 + (self.room_number - 1)
        enemy_damage = 5 + self.room_number
        for _ in range(3):
            pos = self._get_random_empty_pos(occupied_positions)
            occupied_positions.add(tuple(pos))

            # Define patrol area
            patrol_w, patrol_h = self.np_random.integers(2, 5, size=2)
            p1 = pos
            p2 = np.array([pos[0] + patrol_w, pos[1]])
            p3 = np.array([pos[0] + patrol_w, pos[1] + patrol_h])
            p4 = np.array([pos[0], pos[1] + patrol_h])
            patrol_points = [p1, p2, p3, p4]
            # Clamp points to be inside the room
            for i, p in enumerate(patrol_points):
                patrol_points[i] = np.clip(p, [1, 1], [self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2])

            self.enemies.append({
                'pos': pos,
                'health': enemy_health,
                'max_health': enemy_health,
                'damage': enemy_damage,
                'patrol_points': patrol_points,
                'patrol_index': 0,
                'damage_flash': 0,
            })

        # Place treasure
        pos = self._get_random_empty_pos(occupied_positions)
        occupied_positions.add(tuple(pos))
        self.treasures.append({'pos': pos, 'collected': False})

    def _get_random_empty_pos(self, occupied):
        while True:
            pos = np.array([
                self.np_random.integers(1, self.GRID_WIDTH - 1),
                self.np_random.integers(1, self.GRID_HEIGHT - 1)
            ])
            if tuple(pos) not in occupied:
                return pos

    def _get_move_vector(self, movement_action):
        if movement_action == 1: return np.array([0, -1])  # Up
        if movement_action == 2: return np.array([0, 1])  # Down
        if movement_action == 3: return np.array([-1, 0])  # Left
        if movement_action == 4: return np.array([1, 0])  # Right
        return np.array([0, 0])  # None

    def _is_walkable(self, pos):
        x, y = pos
        return 1 <= x < self.GRID_WIDTH - 1 and 1 <= y < self.GRID_HEIGHT - 1

    def _get_dist_to_nearest_target(self):
        targets = []
        for e in self.enemies:
            targets.append(e['pos'])
        for t in self.treasures:
            if not t['collected']:
                targets.append(t['pos'])

        if not targets:
            return None

        distances = [np.linalg.norm(self.player_pos - t_pos) for t_pos in targets]
        return min(distances)

    def _player_attack(self):
        reward = 0
        attack_pos = self.player_pos + self.player_facing_dir
        self._add_particles(self._grid_to_pixel(attack_pos), self.COLOR_PLAYER_ACCENT, 15, 'slash')

        for enemy in self.enemies:
            if np.array_equal(enemy['pos'], attack_pos):
                damage = 10
                enemy['health'] -= damage
                enemy['damage_flash'] = 5
                self.score += 1
                reward += 1
                self._add_floating_text(f"-{damage}", self._grid_to_pixel(enemy['pos']), self.COLOR_TEXT)

                if enemy['health'] <= 0:
                    self.score += 5
                    reward += 5
                    self._add_particles(self._grid_to_pixel(enemy['pos']), self.COLOR_ENEMY, 30, 'burst')

        self.enemies = [e for e in self.enemies if e['health'] > 0]
        return reward

    def _get_exit_pos(self):
        return np.array([self.GRID_WIDTH // 2, 0])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if x == 0 or x == self.GRID_WIDTH - 1 or y == 0 or y == self.GRID_HEIGHT - 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw exit if available
        if not self.enemies:
            exit_pos = self._get_exit_pos()
            rect = pygame.Rect(exit_pos[0] * self.TILE_SIZE, exit_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.TILE_SIZE // 4, self.COLOR_BG)

        # Draw treasures
        for treasure in self.treasures:
            if not treasure['collected']:
                rect = pygame.Rect(treasure['pos'][0] * self.TILE_SIZE, treasure['pos'][1] * self.TILE_SIZE,
                                    self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_TREASURE, rect.inflate(-8, -8))

        # Draw enemies
        for enemy in self.enemies:
            center_px = self._grid_to_pixel(enemy['pos'])
            size = int(self.TILE_SIZE * 0.8)
            rect = pygame.Rect(center_px[0] - size // 2, center_px[1] - size // 2, size, size)

            color = self.COLOR_ENEMY_ACCENT if enemy['damage_flash'] > 0 else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

            # Health bar
            self._render_health_bar(center_px, enemy['health'], enemy['max_health'])

        # Draw player
        center_px = self._grid_to_pixel(self.player_pos)
        size = int(self.TILE_SIZE * 0.9)
        rect = pygame.Rect(center_px[0] - size // 2, center_px[1] - size // 2, size, size)

        color = self.COLOR_PLAYER_ACCENT if self.player_damage_flash > 0 else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Player facing indicator
        eye_pos = (center_px[0] + self.player_facing_dir[0] * size * 0.25,
                   center_px[1] + self.player_facing_dir[1] * size * 0.25)
        pygame.draw.circle(self.screen, self.COLOR_BG, (int(eye_pos[0]), int(eye_pos[1])), 3)

        # Health bar
        self._render_health_bar(center_px, self.player_health, self.player_max_health, self.COLOR_HEALTH_BAR_FILL)

    def _render_health_bar(self, center_px, current, maximum, color=(200, 0, 0)):
        if current < maximum:
            bar_w = self.TILE_SIZE * 0.8
            bar_h = 5
            bar_y = center_px[1] - self.TILE_SIZE // 2 - 8
            bg_rect = pygame.Rect(center_px[0] - bar_w // 2, bar_y, bar_w, bar_h)

            fill_w = (current / maximum) * bar_w
            fill_rect = pygame.Rect(center_px[0] - bar_w // 2, bar_y, fill_w, bar_h)

            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=2)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=2)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        # Floating text
        for ft in self.floating_texts:
            text_surf = ft['surf']
            text_surf.set_alpha(ft['alpha'])
            self.screen.blit(text_surf, (
            ft['pos'][0] - text_surf.get_width() // 2, ft['pos'][1] - text_surf.get_height() // 2))

    def _update_effects(self):
        # Update flashes
        if self.player_damage_flash > 0: self.player_damage_flash -= 1
        for enemy in self.enemies:
            if enemy['damage_flash'] > 0: enemy['damage_flash'] -= 1

        # Update particles
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['radius'] -= 0.2
            if p['radius'] > 0:
                new_particles.append(p)
        self.particles = new_particles

        # Update floating texts
        new_texts = []
        for ft in self.floating_texts:
            ft['pos'][1] -= 1
            ft['age'] += 1
            ft['alpha'] = max(0, 255 * (1 - ft['age'] / ft['lifespan']))
            if ft['age'] < ft['lifespan']:
                new_texts.append(ft)
        self.floating_texts = new_texts

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"Health: {self.player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Room
        room_text = self.font_ui.render(f"Room: {self.room_number} / {self.NUM_ROOMS}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (self.WIDTH - room_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "room": self.room_number,
        }

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        return np.array([x, y])

    def _add_particles(self, pos, color, count, style='burst'):
        for _ in range(count):
            if style == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            elif style == 'impact':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            elif style == 'slash':
                base_angle = math.atan2(self.player_facing_dir[1], self.player_facing_dir[0])
                angle = base_angle + self.np_random.uniform(-math.pi / 4, math.pi / 4)
                speed = self.np_random.uniform(3, 6)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

            self.particles.append({
                'pos': pos.copy().astype(np.float64),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': color,
            })

    def _add_floating_text(self, text, pos, color, lifespan=30, size=20):
        font = pygame.font.Font(None, size)
        self.floating_texts.append({
            'surf': font.render(text, True, color),
            'pos': list(pos),
            'age': 0,
            'lifespan': lifespan,
            'alpha': 255,
        })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here.
    # We are manually handling rendering and event processing.
    env = GameEnv()
    obs, info = env.reset()

    running = True
    game_over_screen = False

    # Set up the display window
    pygame.display.set_caption("Dungeon Explorer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = env.action_space.sample()
    action.fill(0)  # Start with no-op

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and env.game_over:
                obs, info = env.reset()
                game_over_screen = False

        if not env.game_over:
            # --- Player Input ---
            keys = pygame.key.get_pressed()
            movement = 0  # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # Since this is turn-based, we only step when a key is pressed.
            # We can check if any relevant key is down.
            action_taken = any([
                keys[pygame.K_UP], keys[pygame.K_w],
                keys[pygame.K_DOWN], keys[pygame.K_s],
                keys[pygame.K_LEFT], keys[pygame.K_a],
                keys[pygame.K_RIGHT], keys[pygame.K_d],
                keys[pygame.K_SPACE]
            ])

            if action_taken:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                print(
                    f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
                # A small delay to make turns visible
                pygame.time.wait(100)

        else:
            if not game_over_screen:
                print("Game Over! Press 'R' to restart.")
                game_over_screen = True

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Limit frame rate for input responsiveness

    env.close()