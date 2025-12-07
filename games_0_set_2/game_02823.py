# Generated: 2025-08-28T06:05:12.748401
# Source Brief: brief_02823.md
# Brief Index: 2823


import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "No action waits a turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a 5-room dungeon, defeat enemies, and reach the golden exit. "
        "Each action is a turn. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 5
    TILE_SIZE = 70
    GAME_AREA_WIDTH = GRID_SIZE * TILE_SIZE
    GAME_AREA_HEIGHT = GRID_SIZE * TILE_SIZE
    MARGIN_X = (SCREEN_WIDTH - GAME_AREA_WIDTH) // 2
    MARGIN_Y = (SCREEN_HEIGHT - GAME_AREA_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_FLOOR = (40, 40, 50)
    COLOR_WALL = (80, 80, 90)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_EXIT = (255, 215, 0)
    COLOR_HEALTH_GREEN = (0, 255, 0)
    COLOR_HEALTH_RED = (255, 0, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_ATTACK_FLASH = (255, 255, 200)

    # Game settings
    MAX_STEPS = 1000
    NUM_ROOMS = 5
    ENEMIES_PER_ROOM = 2
    PLAYER_MAX_HEALTH = 100
    ENEMY_MAX_HEALTH = 20
    ENEMY_DAMAGE = 10
    PLAYER_DAMAGE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_damage = pygame.font.Font(None, 20)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.player_pos = [0, 0]
        self.last_move_direction = (0, -1)  # Default up
        self.current_room_index = 0
        self.rooms = []
        self.vfx = []

        # This will be properly seeded in reset()
        self.np_random = np.random.default_rng()

        # FIX: Call reset to initialize the game state (e.g., self.rooms)
        # before any methods that rely on it (like _get_observation) are called.
        self.reset()
        self.validate_implementation()

    def _generate_dungeon(self):
        self.rooms = []
        total_enemies = 0
        for i in range(self.NUM_ROOMS):
            # Layout: 0=floor, 1=wall, 2=door_up, 3=door_down, 4=exit
            layout = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            layout[1:-1, 1:-1] = 0  # Hollow out the center for the floor

            # Add doors
            if i > 0:  # Entry door
                layout[self.GRID_SIZE - 1, self.GRID_SIZE // 2] = 3
            if i < self.NUM_ROOMS - 1:  # Exit door
                layout[0, self.GRID_SIZE // 2] = 2

            # Add final exit
            if i == self.NUM_ROOMS - 1:
                layout[0, self.GRID_SIZE // 2] = 4

            # Add enemies
            enemies = []
            for _ in range(self.ENEMIES_PER_ROOM):
                if total_enemies >= 10: break

                # Find valid spawn points
                valid_spawns = []
                for r in range(1, self.GRID_SIZE - 1):
                    for c in range(1, self.GRID_SIZE - 1):
                        is_occupied = False
                        for e in enemies:
                            if e['pos'] == [c, r]:
                                is_occupied = True
                                break
                        if not is_occupied:
                            valid_spawns.append([c, r])

                if not valid_spawns: continue

                pos = list(self.np_random.choice(valid_spawns, axis=0))

                # Simple patrol logic
                patrol_target_1 = list(self.np_random.choice(valid_spawns, axis=0))
                patrol_target_2 = list(self.np_random.choice(valid_spawns, axis=0))

                enemies.append({
                    'pos': pos,
                    'health': self.ENEMY_MAX_HEALTH,
                    'patrol_1': patrol_target_1,
                    'patrol_2': patrol_target_2,
                    'patrol_target': patrol_target_2
                })
                total_enemies += 1

            self.rooms.append({'layout': layout, 'enemies': enemies})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        self.current_room_index = 0
        self.last_move_direction = (0, -1)
        self.vfx = []

        self._generate_dungeon()
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 2]  # Start near bottom of first room

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1

        reward = -0.01  # Small penalty for each turn to encourage efficiency

        # --- Player Turn ---
        if space_held:
            # Attack action
            target_pos = [self.player_pos[0] + self.last_move_direction[0],
                          self.player_pos[1] + self.last_move_direction[1]]

            # Sound effect placeholder: player_attack_swing.wav
            self.vfx.append({'type': 'attack_swing', 'pos': self.player_pos, 'dir': self.last_move_direction, 'timer': 5})

            attacked_enemy = False
            for enemy in self.rooms[self.current_room_index]['enemies']:
                if enemy['health'] > 0 and enemy['pos'] == target_pos:
                    damage = self.PLAYER_DAMAGE
                    enemy['health'] -= damage
                    reward += damage  # Reward for dealing damage
                    self.score += damage

                    self._add_damage_vfx(target_pos, damage, self.COLOR_PLAYER)
                    # Sound effect placeholder: enemy_hit.wav

                    if enemy['health'] <= 0:
                        # Sound effect placeholder: enemy_die.wav
                        self._add_death_vfx(enemy['pos'])
                    attacked_enemy = True
                    break
        else:
            # Movement or Wait action
            if movement != 0:
                move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
                self.last_move_direction = move_dir

                next_pos = [self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1]]

                tile_type = self.rooms[self.current_room_index]['layout'][next_pos[1], next_pos[0]]

                is_enemy_at_next_pos = any(
                    e['pos'] == next_pos and e['health'] > 0 for e in self.rooms[self.current_room_index]['enemies'])

                if tile_type != 1 and not is_enemy_at_next_pos:  # Not a wall and not an enemy
                    self.player_pos = next_pos
                    # Sound effect placeholder: player_step.wav

                    # Handle room transitions
                    if tile_type == 2 and self.current_room_index < self.NUM_ROOMS - 1:  # Door up
                        self.current_room_index += 1
                        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 2]
                    elif tile_type == 3 and self.current_room_index > 0:  # Door down
                        self.current_room_index -= 1
                        self.player_pos = [self.GRID_SIZE // 2, 1]

        # --- Enemy Turn ---
        for enemy in self.rooms[self.current_room_index]['enemies']:
            if enemy['health'] <= 0:
                continue

            # Check for adjacency to player
            dx = abs(enemy['pos'][0] - self.player_pos[0])
            dy = abs(enemy['pos'][1] - self.player_pos[1])

            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                # Attack player
                self.player_health -= self.ENEMY_DAMAGE
                reward -= 0.1 * self.ENEMY_DAMAGE  # Penalty for taking damage
                self._add_damage_vfx(self.player_pos, self.ENEMY_DAMAGE, self.COLOR_ENEMY)
                # Sound effect placeholder: player_hit.wav
            else:
                # Move via patrol
                target = enemy['patrol_target']
                if enemy['pos'] == target:
                    enemy['patrol_target'] = enemy['patrol_1'] if target == enemy['patrol_2'] else enemy['patrol_2']
                    target = enemy['patrol_target']

                ex, ey = enemy['pos']
                tx, ty = target

                move_x, move_y = np.sign(tx - ex), np.sign(ty - ey)

                # Try moving horizontally first
                new_pos_x = [ex + move_x, ey]
                if move_x != 0 and self.rooms[self.current_room_index]['layout'][
                    new_pos_x[1], new_pos_x[0]] == 0 and new_pos_x != self.player_pos:
                    enemy['pos'] = new_pos_x
                # Then try vertically
                elif move_y != 0:
                    new_pos_y = [ex, ey + move_y]
                    if self.rooms[self.current_room_index]['layout'][
                        new_pos_y[1], new_pos_y[0]] == 0 and new_pos_y != self.player_pos:
                        enemy['pos'] = new_pos_y

        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            if self.player_health <= 0:
                reward = -100.0
                self.game_over = True
            elif self.rooms[self.current_room_index]['layout'][self.player_pos[1], self.player_pos[0]] == 4:
                reward = 100.0
                self.game_over = True

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if self.rooms[self.current_room_index]['layout'][self.player_pos[1], self.player_pos[0]] == 4:
            return True
        # Truncation is handled separately now
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "room": self.current_room_index + 1
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._update_and_render_vfx()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        current_room = self.rooms[self.current_room_index]
        layout = current_room['layout']

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(self.MARGIN_X + c * self.TILE_SIZE, self.MARGIN_Y + r * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                tile_type = layout[r, c]

                color = self.COLOR_FLOOR
                if tile_type == 1: color = self.COLOR_WALL
                elif tile_type == 4: color = self.COLOR_EXIT

                pygame.draw.rect(self.screen, color, rect)
                if tile_type == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Outline walls
                elif tile_type in [2, 3]:  # Doors
                    pygame.draw.rect(self.screen, (90, 70, 50), rect.inflate(-10, -10))

        # Draw enemies
        for enemy in current_room['enemies']:
            if enemy['health'] > 0:
                self._draw_entity(enemy['pos'], self.COLOR_ENEMY, enemy['health'], self.ENEMY_MAX_HEALTH)

        # Draw player
        self._draw_entity(self.player_pos, self.COLOR_PLAYER, self.player_health, self.PLAYER_MAX_HEALTH)

    def _draw_entity(self, pos, color, health, max_health):
        rect = pygame.Rect(
            self.MARGIN_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE * 0.15,
            self.MARGIN_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE * 0.15,
            self.TILE_SIZE * 0.7,
            self.TILE_SIZE * 0.7
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Health bar
        if health < max_health:
            bar_width = self.TILE_SIZE * 0.7
            bar_height = 5
            bar_x = self.MARGIN_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE * 0.15
            bar_y = self.MARGIN_Y + pos[1] * self.TILE_SIZE

            health_ratio = max(0, health / max_health)

            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN,
                             (bar_x, bar_y, bar_width * health_ratio, bar_height))

    def _render_ui(self):
        # Player Health
        health_text = self.font_ui.render(f"Health: {max(0, self.player_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Room Number
        room_text = self.font_ui.render(f"Room: {self.current_room_index + 1}/{self.NUM_ROOMS}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (self.SCREEN_WIDTH - room_text.get_width() - 10, 10))

    def _add_damage_vfx(self, pos, amount, color):
        vfx_pos = (
            self.MARGIN_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2,
            self.MARGIN_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        )
        self.vfx.append({'type': 'damage_text', 'pos': vfx_pos, 'text': str(amount), 'color': color, 'timer': 20})

    def _add_death_vfx(self, pos):
        center_x = self.MARGIN_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.MARGIN_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.vfx.append({
                'type': 'particle',
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'timer': self.np_random.integers(15, 30),
                'color': self.COLOR_ENEMY
            })

    def _update_and_render_vfx(self):
        active_vfx = []
        for vfx in self.vfx:
            vfx['timer'] -= 1
            if vfx['timer'] > 0:
                active_vfx.append(vfx)

                if vfx['type'] == 'attack_swing':
                    start_pos = (
                        self.MARGIN_X + vfx['pos'][0] * self.TILE_SIZE + self.TILE_SIZE // 2,
                        self.MARGIN_Y + vfx['pos'][1] * self.TILE_SIZE + self.TILE_SIZE // 2
                    )
                    end_pos = (
                        start_pos[0] + vfx['dir'][0] * self.TILE_SIZE * 0.7,
                        start_pos[1] + vfx['dir'][1] * self.TILE_SIZE * 0.7
                    )
                    # Create a temporary surface for alpha blending
                    line_surf = self.screen.copy()
                    line_surf.set_colorkey(self.COLOR_BG)
                    line_surf.set_alpha(int(255 * (vfx['timer'] / 5)))
                    pygame.draw.line(line_surf, self.COLOR_ATTACK_FLASH, start_pos, end_pos, 3)
                    self.screen.blit(line_surf, (0, 0))


                elif vfx['type'] == 'damage_text':
                    text_surf = self.font_damage.render(vfx['text'], True, vfx['color'])
                    alpha = int(255 * (vfx['timer'] / 20))
                    text_surf.set_alpha(alpha)
                    pos = (vfx['pos'][0] - text_surf.get_width() // 2, vfx['pos'][1] - (20 - vfx['timer']))
                    self.screen.blit(text_surf, pos)

                elif vfx['type'] == 'particle':
                    vfx['pos'][0] += vfx['vel'][0]
                    vfx['pos'][1] += vfx['vel'][1]
                    vfx['vel'][1] += 0.1  # Gravity
                    alpha = int(255 * (vfx['timer'] / 30))
                    color = (*vfx['color'], alpha)
                    if 0 <= vfx['pos'][0] < self.SCREEN_WIDTH and 0 <= vfx['pos'][1] < self.SCREEN_HEIGHT:
                        try:
                           pygame.gfxdraw.pixel(self.screen, int(vfx['pos'][0]), int(vfx['pos'][1]), color)
                        except (ValueError, TypeError):
                           # This can happen if alpha is out of range, ignore for robustness
                           pass
        self.vfx = active_vfx

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a display, so it won't run in a pure headless environment
    try:
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]

        env = GameEnv()
        obs, info = env.reset(seed=random.randint(0, 10000))
        done = False

        # Create a window to display the game
        pygame.display.set_caption("Dungeon Crawler")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        print(GameEnv.game_description)
        print(GameEnv.user_guide)

        last_action_time = pygame.time.get_ticks()
        action_cooldown = 200 # ms

        while not done:
            # --- Human Input ---
            movement_action = 0  # no-op
            space_action = 0
            action_to_take = None

            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    # Set action_to_take to None to prevent processing a final action
                    action_to_take = None
                    break
            if done:
                break

            current_time = pygame.time.get_ticks()
            if current_time - last_action_time > action_cooldown:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement_action = 1
                elif keys[pygame.K_DOWN]: movement_action = 2
                elif keys[pygame.K_LEFT]: movement_action = 3
                elif keys[pygame.K_RIGHT]: movement_action = 4

                if keys[pygame.K_SPACE]: space_action = 1

                if movement_action != 0 or space_action != 0:
                    action_to_take = (movement_action, space_action, 0)
                    last_action_time = current_time


            if action_to_take is not None:
                obs, reward, terminated, truncated, info = env.step(action_to_take)
                done = terminated or truncated
                print(
                    f"Step: {info['steps']}, Room: {info['room']}, Health: {info['player_health']}, Reward: {reward:.2f}, Score: {info['score']}")

            # --- Rendering ---
            # The observation is already a rendered frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))

            pygame.display.flip()

            clock.tick(60) # Limit frame rate

        print("Game Over!")
        env.close()

    except pygame.error as e:
        print(f"Pygame error (likely due to headless environment): {e}")
        print("Skipping interactive test.")
        # Run a non-interactive test to confirm the fix
        try:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            env = GameEnv()
            print("Headless environment created and validated successfully.")
            env.close()
        except Exception as e:
            print(f"Error during headless test: {e}")