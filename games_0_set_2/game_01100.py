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
        "Controls: Arrow keys to move on the grid. Press Space to attack adjacent squares."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a fast-paced, grid-based arena by strategically moving and attacking."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    # Colors
    COLOR_BG = (18, 18, 28)
    COLOR_GRID = (30, 30, 45)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_ATTACK = (255, 255, 255)
    COLOR_MONSTER_1 = (255, 80, 80)
    COLOR_MONSTER_2 = (255, 140, 80)
    COLOR_MONSTER_3 = (200, 80, 255)
    COLOR_POTION = (80, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 40, 60)
    COLOR_HEALTH_BAR = (80, 220, 80)
    COLOR_HEALTH_BAR_BG = (180, 50, 50)

    # Game parameters
    PLAYER_MAX_HEALTH = 10
    PLAYER_MOVE_COOLDOWN = 6  # frames
    PLAYER_ATTACK_COOLDOWN = 15  # frames
    MAX_EPISODE_STEPS = 1000
    MONSTERS_TO_CLEAR_STAGE = 10
    POTION_SPAWN_CHANCE = 0.008
    POTION_HEAL_AMOUNT = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_grid_pos = None
        self.player_health = None
        self.player_move_cd = None
        self.player_attack_cd = None
        self.monsters = None
        self.potions = None
        self.particles = None
        self.steps = None
        self.score = None
        self.stage = None
        self.monsters_defeated_in_stage = None
        self.game_over = None
        self.rng = None

        # self.reset() is called here to initialize state for validation
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use a numpy random number generator for consistency
        self.rng = np.random.default_rng(seed)

        # Player state
        self.player_grid_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_move_cd = 0
        self.player_attack_cd = 0

        # Game state
        self.monsters = []
        self.potions = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.monsters_defeated_in_stage = 0
        self.game_over = False

        # Initial monster spawn
        for _ in range(3):
            self._spawn_monster()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Cooldowns ---
        if self.player_move_cd > 0: self.player_move_cd -= 1
        if self.player_attack_cd > 0: self.player_attack_cd -= 1

        # --- Handle Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Player Movement
        prev_grid_pos = self.player_grid_pos
        if self.player_move_cd == 0:
            target_x, target_y = self.player_grid_pos
            if movement == 1: target_y -= 1  # Up
            elif movement == 2: target_y += 1  # Down
            elif movement == 3: target_x -= 1  # Left
            elif movement == 4: target_x += 1  # Right

            if movement != 0:
                self.player_grid_pos = (
                    max(0, min(self.GRID_WIDTH - 1, target_x)),
                    max(0, min(self.GRID_HEIGHT - 1, target_y))
                )
                if self.player_grid_pos != prev_grid_pos:
                    self.player_move_cd = self.PLAYER_MOVE_COOLDOWN
                    # Small penalty for moving
                    if not self._is_adjacent_to_monster(self.player_grid_pos):
                        reward -= 0.01

        # Player Attack
        if space_held and self.player_attack_cd == 0:
            self.player_attack_cd = self.PLAYER_ATTACK_COOLDOWN
            attack_positions = self._get_adjacent_cells(self.player_grid_pos)
            self._create_attack_particles(attack_positions)

            for monster in self.monsters[:]:
                if monster['grid_pos'] in attack_positions:
                    monster['health'] -= 1
                    reward += 1.0  # Reward for damaging
                    self._create_hit_particles(monster['pos'])
                    if monster['health'] <= 0:
                        self.score += 10
                        reward += 5.0  # Reward for defeating
                        self.monsters_defeated_in_stage += 1
                        self.monsters.remove(monster)
                        self._create_death_particles(monster['pos'])

        # --- Update Game Logic ---
        reward += self._update_monsters()
        self._update_player_position()
        self._update_particles()
        reward += self._check_potion_collection()

        # Randomly spawn potions
        if self.rng.random() < self.POTION_SPAWN_CHANCE and len(self.potions) < 3:
            self._spawn_potion()

        # --- Stage Progression ---
        if self.monsters_defeated_in_stage >= self.MONSTERS_TO_CLEAR_STAGE:
            self._next_stage()
            reward += 50.0

        # --- Termination ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 50.0
            terminated = True
            self.game_over = True

        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            float(np.clip(reward, -100, 100)),
            terminated,
            truncated,
            self._get_info()
        )

    def _next_stage(self):
        self.stage += 1
        self.monsters_defeated_in_stage = 0
        self.score += 100  # Stage clear bonus
        self.potions.clear()
        # Spawn new wave of monsters
        for _ in range(3 + self.stage):
            self._spawn_monster()

    def _update_monsters(self):
        reward_change = 0.0
        for m in self.monsters:
            if m['move_cd'] > 0:
                m['move_cd'] -= 1
            else:
                m['move_cd'] = m['move_speed']
                dx = self.player_grid_pos[0] - m['grid_pos'][0]
                dy = self.player_grid_pos[1] - m['grid_pos'][1]

                if abs(dx) + abs(dy) == 1:  # Adjacent, so attack
                    if m['attack_cd'] == 0:
                        m['attack_cd'] = m['attack_speed']
                        self.player_health -= m['damage']
                        reward_change -= 1.0  # Penalty for taking damage
                        self._create_hit_particles(self.player_pos, self.COLOR_PLAYER)
                    continue

                # Move towards player
                new_gx, new_gy = m['grid_pos']
                if abs(dx) > abs(dy):
                    new_gx += np.sign(dx)
                elif abs(dy) > 0:
                    new_gy += np.sign(dy)

                # Avoid collision with other monsters
                if not any(other['grid_pos'] == (new_gx, new_gy) for other in self.monsters):
                    m['grid_pos'] = (new_gx, new_gy)

            if m['attack_cd'] > 0:
                m['attack_cd'] -= 1

            # Smooth visual movement
            target_pixel_pos = self._grid_to_pixel(m['grid_pos'])
            m['pos'] = m['pos'].lerp(target_pixel_pos, 0.2)
        return reward_change

    def _check_potion_collection(self):
        reward_change = 0.0
        for potion_pos in self.potions[:]:
            if self._grid_to_pixel(self.player_grid_pos).distance_to(potion_pos) < self.CELL_WIDTH / 2:
                self.potions.remove(potion_pos)
                self.player_health = min(self.PLAYER_MAX_HEALTH, self.player_health + self.POTION_HEAL_AMOUNT)
                reward_change += 2.0  # Reward for potion
                self.score += 5
        return reward_change

    def _update_player_position(self):
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_pos = self.player_pos.lerp(target_pixel_pos, 0.4)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_monster(self):
        side = self.rng.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            pos = (self.rng.integers(0, self.GRID_WIDTH), 0)
        elif side == 'bottom':
            pos = (self.rng.integers(0, self.GRID_WIDTH), self.GRID_HEIGHT - 1)
        elif side == 'left':
            pos = (0, self.rng.integers(0, self.GRID_HEIGHT))
        else:  # right
            pos = (self.GRID_WIDTH - 1, self.rng.integers(0, self.GRID_HEIGHT))

        monster_type = self.rng.integers(1, min(self.stage, 3) + 1)
        if monster_type == 1:
            health, damage, color, move_speed, attack_speed = 1, 1, self.COLOR_MONSTER_1, 30, 60
        elif monster_type == 2:
            health, damage, color, move_speed, attack_speed = 2, 1, self.COLOR_MONSTER_2, 25, 50
        else:
            health, damage, color, move_speed, attack_speed = 1, 2, self.COLOR_MONSTER_3, 20, 40

        self.monsters.append({
            'grid_pos': pos,
            'pos': self._grid_to_pixel(pos),
            'health': health,
            'damage': damage,
            'color': color,
            'type': monster_type,
            'move_speed': move_speed,
            'attack_speed': attack_speed,
            'move_cd': 0,
            'attack_cd': 0
        })

    def _spawn_potion(self):
        pos = (self.rng.integers(1, self.GRID_WIDTH - 1), self.rng.integers(1, self.GRID_HEIGHT - 1))
        # Ensure it doesn't spawn on player or monster
        if pos == self.player_grid_pos or any(m['grid_pos'] == pos for m in self.monsters):
            return
        self.potions.append(self._grid_to_pixel(pos))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_potions()
        self._render_monsters()
        self._render_player()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "health": self.player_health,
            "monsters_defeated": self.monsters_defeated_in_stage
        }

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        y = grid_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        return pygame.Vector2(x, y)

    def _is_adjacent_to_monster(self, grid_pos):
        for m in self.monsters:
            if abs(m['grid_pos'][0] - grid_pos[0]) + abs(m['grid_pos'][1] - grid_pos[1]) == 1:
                return True
        return False

    def _get_adjacent_cells(self, grid_pos):
        x, y = grid_pos
        return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _render_player(self):
        size = self.CELL_WIDTH * 0.7
        player_rect = pygame.Rect(self.player_pos.x - size / 2, self.player_pos.y - size / 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        if self.player_attack_cd > self.PLAYER_ATTACK_COOLDOWN - 4:
            for pos in self._get_adjacent_cells(self.player_grid_pos):
                if 0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT:
                    pixel_pos = self._grid_to_pixel(pos)
                    attack_rect = pygame.Rect(pixel_pos.x - self.CELL_WIDTH / 2,
                                              pixel_pos.y - self.CELL_HEIGHT / 2, self.CELL_WIDTH, self.CELL_HEIGHT)
                    s = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
                    alpha = 150 * (self.player_attack_cd / self.PLAYER_ATTACK_COOLDOWN)
                    pygame.draw.rect(s, (*self.COLOR_PLAYER_ATTACK, alpha), s.get_rect(), border_radius=5)
                    self.screen.blit(s, attack_rect.topleft)

    def _render_monsters(self):
        size = self.CELL_WIDTH * 0.7
        for m in self.monsters:
            monster_rect = pygame.Rect(m['pos'].x - size / 2, m['pos'].y - size / 2, size, size)
            pygame.draw.rect(self.screen, m['color'], monster_rect, border_radius=4)

    def _render_potions(self):
        for pos in self.potions:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(self.CELL_WIDTH * 0.2 + pulse * self.CELL_WIDTH * 0.1)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_POTION)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_POTION)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

    def _render_ui(self):
        bar_width, bar_height = 150, 20
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (12, 12, bar_width - 4, bar_height - 4),
                         border_radius=4)
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR,
                             (12, 12, (bar_width - 4) * health_ratio, bar_height - 4), border_radius=4)

        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        stage_text = self.font_large.render(f"STAGE: {self.stage}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH / 2 - stage_text.get_width() / 2, 10))

        monsters_text = self.font_small.render(f"Defeated: {self.monsters_defeated_in_stage}/{self.MONSTERS_TO_CLEAR_STAGE}",
                                               True, self.COLOR_UI_TEXT)
        self.screen.blit(monsters_text, (self.SCREEN_WIDTH / 2 - monsters_text.get_width() / 2, 45))

    def _create_particles(self, pos, count, color, min_life, max_life, min_speed, max_speed, size):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(min_speed, max_speed)
            life = self.rng.integers(min_life, max_life + 1)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'size': size
            })

    def _create_hit_particles(self, pos, color=(255, 255, 255)):
        self._create_particles(pos, 5, color, 5, 10, 1, 3, 2)

    def _create_death_particles(self, pos):
        self._create_particles(pos, 20, self.COLOR_MONSTER_1, 15, 30, 2, 5, 3)

    def _create_attack_particles(self, positions):
        pass

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        self.reset()
        self.player_health = 1
        self.potions.append(self._grid_to_pixel(self.player_grid_pos))
        self._check_potion_collection()
        assert self.player_health <= self.PLAYER_MAX_HEALTH

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()

    # To run with display, unset the dummy video driver
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Arena")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30)

    env.close()