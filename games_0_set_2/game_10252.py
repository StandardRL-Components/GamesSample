import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:05:47.540091
# Source Brief: brief_00252.md
# Brief Index: 252
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth/puzzle Gymnasium environment.

    The player must craft keys at stations while summoning emotion-matched
    allies to distract patrolling dream guardians. The goal is to unlock
    all 5 dream chambers without being caught.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Craft keys at stations and summon allies to distract guardians. "
        "Unlock all dream chambers without being caught to win."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to craft keys at stations or summon allies. "
        "Press shift to cycle through emotions for your allies."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_CELL_SIZE = 32
    GRID_WIDTH = SCREEN_WIDTH // GRID_CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_CELL_SIZE

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_GUARDIAN = (180, 50, 255)
    COLOR_DOOR = (255, 215, 0)
    COLOR_STATION = (0, 200, 200)
    EMOTION_COLORS = [
        (255, 50, 50),   # 0: Anger (Red)
        (50, 100, 255),  # 1: Sadness (Blue)
        (50, 255, 50),   # 2: Joy (Green)
        (255, 255, 50),  # 3: Fear (Yellow)
    ]
    EMOTION_NAMES = ["ANGER", "SADNESS", "JOY", "FEAR"]
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Game Parameters
    MAX_STEPS = 2000
    NUM_CHAMBERS = 5
    ALLY_LIFESPAN = 20
    GUARDIAN_PAUSE_DURATION = 10
    CRAFTING_TIME = 15
    BASE_GUARDIAN_SPEED = 0.1 # tiles per step
    
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)

        self._define_chambers()
        
        # State variables are initialized in reset()
        self.player_pos = [0, 0]
        self.guardians = []
        self.allies = []
        self.particles = []
        self.unlocked_chambers = 0
        self.current_chamber_level = 0
        self.current_emotion_idx = 0
        self.crafting_progress = 0
        self.key_crafted = False
        self.prev_shift_state = False
        self.prev_space_state = False
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()

    def _define_chambers(self):
        self.chambers = [
            # Chamber 0
            {
                'walls': [(x, 0) for x in range(20)] + [(x, 11) for x in range(20)] + [(0, y) for y in range(1, 11)] + [(19, y) for y in range(1, 11)] + [(10,y) for y in range(3,9)],
                'start_pos': [2, 5], 'station_pos': [17, 5], 'door_pos': [10, 2],
                'guardians': [{'path': [[5, 2], [5, 9]]}]
            },
            # Chamber 1
            {
                'walls': [(x, 0) for x in range(20)] + [(x, 11) for x in range(20)] + [(0, y) for y in range(1, 11)] + [(19, y) for y in range(1, 11)],
                'start_pos': [2, 2], 'station_pos': [17, 9], 'door_pos': [10, 0],
                'guardians': [{'path': [[5, 2], [5, 9], [15, 9], [15, 2]]}]
            },
            # Chamber 2
            {
                'walls': [(x, 0) for x in range(20)] + [(x, 11) for x in range(20)] + [(0, y) for y in range(1, 11)] + [(19, y) for y in range(1, 11)] + [(x, 5) for x in range(4, 16)],
                'start_pos': [2, 2], 'station_pos': [17, 8], 'door_pos': [10, 11],
                'guardians': [{'path': [[2, 7], [8, 7]]}, {'path': [[17, 3], [11, 3]]}]
            },
            # Chamber 3
            {
                'walls': [(x, 0) for x in range(20)] + [(x, 11) for x in range(20)] + [(0, y) for y in range(1, 11)] + [(19, y) for y in range(1, 11)] + [(5,y) for y in range(0,7)] + [(14,y) for y in range(5,12)],
                'start_pos': [2, 5], 'station_pos': [17, 2], 'door_pos': [5, 7],
                'guardians': [{'path': [[3, 2], [12, 2], [12, 9], [3, 9]]}, {'path': [[8, 4], [8, 7]]}]
            },
            # Chamber 4
            {
                'walls': [(x, 0) for x in range(20)] + [(x, 11) for x in range(20)] + [(0, y) for y in range(1, 11)] + [(19, y) for y in range(1, 11)] + [(x,3) for x in range(0,15)] + [(x,8) for x in range(5,20)],
                'start_pos': [2, 1], 'station_pos': [17, 10], 'door_pos': [15, 3],
                'guardians': [{'path': [[2, 5], [17, 5]]}, {'path': [[10, 1], [10, 10]]}]
            }
        ]

    def _setup_chamber(self, level):
        self.current_chamber_level = level
        chamber_data = self.chambers[level]
        
        self.player_pos = list(chamber_data['start_pos'])
        self.station_pos = chamber_data['station_pos']
        self.door_pos = chamber_data['door_pos']
        self.walls = set(chamber_data['walls'])
        
        self.key_crafted = False
        self.crafting_progress = 0
        
        self.guardians = []
        for g_data in chamber_data['guardians']:
            start_node = g_data['path'][0]
            self.guardians.append({
                'pos': [float(start_node[0]), float(start_node[1])],
                'path': g_data['path'], 'path_index': 0, 'paused_timer': 0, 'forward': True
            })
        
        self.allies = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.unlocked_chambers = 0
        self.current_emotion_idx = 0
        self.prev_shift_state = False
        self.prev_space_state = False
        
        self._setup_chamber(0)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for taking time

        reward += self._handle_actions(action)
        self._update_allies()
        self._update_guardians()
        self._update_particles()
        reward += self._check_collisions_and_events()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.unlocked_chambers >= self.NUM_CHAMBERS and not terminated:
            reward += 100 # Victory reward
            terminated = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1
        reward = 0

        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if tuple(new_pos) not in self.walls:
                self.player_pos = new_pos
        
        if shift_held and not self.prev_shift_state:
            self.current_emotion_idx = (self.current_emotion_idx + 1) % len(self.EMOTION_COLORS)
        self.prev_shift_state = shift_held

        is_at_station = self.player_pos == self.station_pos
        if space_held:
            if is_at_station and not self.key_crafted:
                self.crafting_progress += 1
                reward += 0.2
                # SFX: Crafting tick
                if self.crafting_progress >= self.CRAFTING_TIME:
                    self.key_crafted = True
                    self.crafting_progress = self.CRAFTING_TIME
                    self._create_particles(self._grid_to_pixel(self.station_pos), self.COLOR_STATION, 20, 3)
                    # SFX: Key Crafted
            elif not is_at_station and not self.prev_space_state:
                self._summon_ally()
                reward += 0.1
        self.prev_space_state = space_held
        
        return reward

    def _update_allies(self):
        for ally in self.allies[:]:
            ally['lifespan'] -= 1
            if ally['lifespan'] <= 0:
                self.allies.remove(ally)
                continue

            if self.guardians:
                nearest_guardian = min(self.guardians, key=lambda g: self._dist(ally['pos'], g['pos']))
                direction = [nearest_guardian['pos'][0] - ally['pos'][0], nearest_guardian['pos'][1] - ally['pos'][1]]
                dist = math.hypot(*direction)
                if dist > 0.5:
                    ally['pos'][0] += (direction[0] / dist) * 0.2 # Ally speed
                    ally['pos'][1] += (direction[1] / dist) * 0.2

    def _update_guardians(self):
        guardian_speed = self.BASE_GUARDIAN_SPEED + (0.05 * self.unlocked_chambers)
        for g in self.guardians:
            if g['paused_timer'] > 0:
                g['paused_timer'] -= 1
                continue
            
            path = g['path']
            if len(path) <= 1: continue

            target_node = path[g['path_index']]
            direction = [target_node[0] - g['pos'][0], target_node[1] - g['pos'][1]]
            dist = math.hypot(*direction)

            if dist < guardian_speed:
                g['pos'] = list(target_node)
                if g['forward']:
                    g['path_index'] += 1
                    if g['path_index'] >= len(path):
                        g['path_index'] = len(path) - 2
                        g['forward'] = False
                else:
                    g['path_index'] -= 1
                    if g['path_index'] < 0:
                        g['path_index'] = 1
                        g['forward'] = True
            elif dist > 0:
                g['pos'][0] += (direction[0] / dist) * guardian_speed
                g['pos'][1] += (direction[1] / dist) * guardian_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_collisions_and_events(self):
        reward = 0
        player_grid_pos = tuple(self.player_pos)
        for g in self.guardians:
            if tuple(map(round, g['pos'])) == player_grid_pos:
                self.game_over = True
                reward -= 100
                # SFX: Player caught
                return reward
        
        for g in self.guardians:
            if g['paused_timer'] > 0: continue
            guardian_grid_pos = tuple(map(round, g['pos']))
            for ally in self.allies:
                if tuple(map(round, ally['pos'])) == guardian_grid_pos:
                    g['paused_timer'] = self.GUARDIAN_PAUSE_DURATION
                    ally['lifespan'] = 0 # Ally is consumed
                    self._create_particles(self._grid_to_pixel(guardian_grid_pos), self.COLOR_GUARDIAN, 30, 4)
                    # SFX: Guardian distracted
                    break
        
        if self.key_crafted and self.player_pos == self.door_pos:
            self.unlocked_chambers += 1
            reward += 5
            # SFX: Chamber unlocked
            if self.unlocked_chambers < self.NUM_CHAMBERS:
                self._setup_chamber(self.unlocked_chambers)
            self._create_particles(self._grid_to_pixel(self.door_pos), self.COLOR_DOOR, 50, 5)

        return reward

    def _summon_ally(self):
        # SFX: Ally summoned
        self.allies.append({
            'pos': [float(self.player_pos[0]), float(self.player_pos[1])],
            'emotion_idx': self.current_emotion_idx, 'lifespan': self.ALLY_LIFESPAN
        })
        self._create_particles(self._grid_to_pixel(self.player_pos), self.EMOTION_COLORS[self.current_emotion_idx], 20, 2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y in self.walls:
            rect = pygame.Rect(x * self.GRID_CELL_SIZE, y * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        self._render_glow_rect(self._grid_to_pixel(self.door_pos), (self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), self.COLOR_DOOR, 10 if self.key_crafted else 3)
        self._render_glow_rect(self._grid_to_pixel(self.station_pos), (self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), self.COLOR_STATION, 5)
        
        if self.player_pos == self.station_pos and not self.key_crafted:
            progress_w = (self.crafting_progress / self.CRAFTING_TIME) * (self.GRID_CELL_SIZE - 4)
            bar_x = self.station_pos[0] * self.GRID_CELL_SIZE + 2
            bar_y = self.station_pos[1] * self.GRID_CELL_SIZE - 8
            pygame.draw.rect(self.screen, self.COLOR_BG, (bar_x, bar_y, self.GRID_CELL_SIZE-4, 5))
            pygame.draw.rect(self.screen, self.COLOR_STATION, (bar_x, bar_y, progress_w, 5))

        for g in self.guardians:
            if len(g['path']) > 1:
                points = [self._grid_to_pixel(p) for p in g['path']]
                pygame.draw.lines(self.screen, (50, 50, 80), False, points, 1)

        for ally in self.allies:
            self._render_glow_circle(self._grid_to_pixel(ally['pos'], is_float=True), 8, self.EMOTION_COLORS[ally['emotion_idx']], 5)
        
        for g in self.guardians:
            color = self.COLOR_GUARDIAN if g['paused_timer'] == 0 else (100, 100, 120)
            self._render_glow_circle(self._grid_to_pixel(g['pos'], is_float=True), 12, color, 8)
            
        self._render_glow_circle(self._grid_to_pixel(self.player_pos), 14, self.COLOR_PLAYER, 10)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _render_ui(self):
        chambers_text = self.font_ui.render(f"Chambers: {self.unlocked_chambers}/{self.NUM_CHAMBERS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(chambers_text, (10, self.SCREEN_HEIGHT - 28))

        emotion_name = self.EMOTION_NAMES[self.current_emotion_idx]
        emotion_color = self.EMOTION_COLORS[self.current_emotion_idx]
        emotion_text = self.font_ui.render(f"Emotion: {emotion_name}", True, emotion_color)
        self.screen.blit(emotion_text, (self.SCREEN_WIDTH - emotion_text.get_width() - 80, self.SCREEN_HEIGHT - 28))
        pygame.draw.rect(self.screen, emotion_color, (self.SCREEN_WIDTH - 70, self.SCREEN_HEIGHT - 28, 60, 20))

        key_status = "READY" if self.key_crafted else "NOT READY"
        key_color = self.COLOR_DOOR if self.key_crafted else (100, 100, 100)
        key_text = self.font_ui.render(f"Key: {key_status}", True, key_color)
        self.screen.blit(key_text, (self.SCREEN_WIDTH // 2 - key_text.get_width() // 2, 10))

    def _render_glow_circle(self, center, radius, color, glow_size):
        center_x, center_y = int(center[0]), int(center[1])
        for i in range(glow_size, 0, -1):
            alpha = int(150 * (1 - i / glow_size))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + i, (*color, alpha))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)

    def _render_glow_rect(self, center, size, color, glow_size):
        rect_x = center[0] - size[0] // 2
        rect_y = center[1] - size[1] // 2
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - i / glow_size))
            glow_surf = pygame.Surface((size[0] + i*2, size[1] + i*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*color, alpha), glow_surf.get_rect(), border_radius=max(1, i // 2))
            self.screen.blit(glow_surf, (rect_x - i, rect_y - i))
        pygame.draw.rect(self.screen, color, (rect_x, rect_y, size[0], size[1]))

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, speed_scale)
            self.particles.append({
                'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color, 'lifespan': random.randint(20, 40), 'size': random.uniform(2, 5)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "unlocked_chambers": self.unlocked_chambers, "key_crafted": self.key_crafted}
    
    def _grid_to_pixel(self, grid_pos, is_float=False):
        offset = self.GRID_CELL_SIZE / 2
        px_pos = [grid_pos[0] * self.GRID_CELL_SIZE + offset, grid_pos[1] * self.GRID_CELL_SIZE + offset]
        return px_pos if is_float else list(map(int, px_pos))

    def _dist(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # The main loop is for manual play and debugging, so we need a real display.
    # We will unset the dummy video driver for this part.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Dream Weaver")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0]
    
    print("\n--- Manual Controls ---")
    print("Arrows/WASD: Move")
    print("Space: Craft (at station) or Summon Ally (elsewhere)")
    print("Shift: Cycle Emotion")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: action[1] = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 0
    
        keys = pygame.key.get_pressed()
        action[0] = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != -0.01:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    print(f"\nGame Over! Final Score: {info['score']:.2f}, Unlocked: {info['unlocked_chambers']}")
    env.close()