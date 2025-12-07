import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:10:49.849581
# Source Brief: brief_00251.md
# Brief Index: 251
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a celestial being through space, collecting starlight fragments while avoiding shadow beasts. "
        "Match pairs of celestial bodies to reveal constellations and score points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to interact with celestial bodies to find matching pairs."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (150, 200, 255)
    COLOR_PLAYER_GLOW = (50, 100, 200)
    COLOR_BEAST = (255, 50, 50)
    COLOR_BEAST_GLOW = (180, 0, 0)
    COLOR_FRAGMENT = (255, 220, 100)
    COLOR_FRAGMENT_GLOW = (200, 150, 0)
    COLOR_CONSTELLATION = (200, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CELESTIAL_BODY_INACTIVE = (80, 80, 120)
    COLOR_CELESTIAL_BODY_SELECTED = (255, 255, 0)
    COLOR_CELESTIAL_BODY_MATCHED = (200, 255, 255)

    # Game Parameters
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    BEAST_RADIUS = 14
    FRAGMENT_RADIUS = 6
    CELESTIAL_RADIUS = 10
    TOUCH_RADIUS = 40
    MIN_SPAWN_DISTANCE = 50

    CELESTIAL_SYMBOLS = [
        # lambda surface, pos, color, radius: pygame.draw.circle(surface, color, pos, int(radius * 0.8), 2),
        lambda surface, pos, color, radius: pygame.draw.rect(surface, color, (pos[0]-radius//2, pos[1]-radius//2, radius, radius), 2),
        lambda surface, pos, color, radius: pygame.draw.polygon(surface, color, [(pos[0], pos[1]-radius*0.8), (pos[0]-radius*0.7, pos[1]+radius*0.4), (pos[0]+radius*0.7, pos[1]+radius*0.4)], 2),
        lambda surface, pos, color, radius: pygame.draw.line(surface, color, (pos[0]-radius*0.7, pos[1]-radius*0.7), (pos[0]+radius*0.7, pos[1]+radius*0.7), 2) or pygame.draw.line(surface, color, (pos[0]-radius*0.7, pos[1]+radius*0.7), (pos[0]+radius*0.7, pos[1]-radius*0.7), 2),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Game state variables initialized in reset()
        self.level = 1
        self.game_over_flag = True # Start with a full reset
        self.player_pos = None
        self.fragments = []
        self.celestial_bodies = []
        self.shadow_beasts = []
        self.particles = []
        self.selected_body_idx = None
        self.steps = 0
        self.score = 0
        self.starfield = []

        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over_flag:
            self.level = 1
            self.score = 0
        
        self.steps = 0
        self.game_over_flag = False
        self.selected_body_idx = None
        self.particles.clear()

        # Generate background starfield
        if not self.starfield:
            for _ in range(200):
                self.starfield.append(
                    (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
                )

        # --- Level Generation ---
        all_entities = []

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.1, self.SCREEN_HEIGHT / 2)
        all_entities.append(self.player_pos)

        # Fragments
        self.fragments = []
        num_fragments = 3 + self.level // 2
        for _ in range(num_fragments):
            self.fragments.append(self._get_valid_spawn_pos(all_entities))
            all_entities.append(self.fragments[-1])

        # Celestial Bodies
        self.celestial_bodies = []
        num_pairs = min(len(self.CELESTIAL_SYMBOLS), 1 + self.level // 3)
        body_types = self.np_random.choice(len(self.CELESTIAL_SYMBOLS), num_pairs, replace=False)
        
        for i in range(num_pairs):
            pos1 = self._get_valid_spawn_pos(all_entities)
            all_entities.append(pos1)
            pos2 = self._get_valid_spawn_pos(all_entities)
            all_entities.append(pos2)
            
            idx1 = len(self.celestial_bodies)
            idx2 = idx1 + 1
            
            self.celestial_bodies.append({'id': idx1, 'pos': pos1, 'type': body_types[i], 'matched': False, 'pair_id': idx2})
            self.celestial_bodies.append({'id': idx2, 'pos': pos2, 'type': body_types[i], 'matched': False, 'pair_id': idx1})
            
        # Shadow Beasts
        self.shadow_beasts = []
        num_beasts = 1 + (self.level -1) // 2
        beast_speed = 1.0 + (self.level - 1) * 0.1
        
        for _ in range(num_beasts):
            path = [self._get_valid_spawn_pos(all_entities, self.SCREEN_WIDTH/2) for _ in range(self.np_random.integers(3, 6))]
            start_pos = path[0]
            self.shadow_beasts.append({
                'pos': pygame.Vector2(start_pos),
                'speed': beast_speed,
                'path': path,
                'target_idx': 1
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Store pre-move state for reward calculation ---
        prev_player_pos = self.player_pos.copy()
        dist_to_fragment_before = self._get_dist_to_nearest(self.player_pos, self.fragments)
        dist_to_beast_before = self._get_dist_to_nearest(self.player_pos, [b['pos'] for b in self.shadow_beasts])

        # --- Player Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

        # --- Continuous Movement Rewards ---
        if move_vec.length() > 0:
            dist_to_fragment_after = self._get_dist_to_nearest(self.player_pos, self.fragments)
            if dist_to_fragment_after < dist_to_fragment_before:
                reward += 0.01
            
            dist_to_beast_after = self._get_dist_to_nearest(self.player_pos, [b['pos'] for b in self.shadow_beasts])
            if dist_to_beast_after > dist_to_beast_before:
                reward += 0.01
            elif dist_to_beast_after < dist_to_beast_before:
                reward -= 0.02 # Penalize moving towards danger more

        # --- Celestial Body Matching ---
        if space_held:
            reward += self._handle_matching()

        # --- Update Game Entities ---
        self._update_shadow_beasts()
        self._update_particles()
        
        # --- Check Collections ---
        collected_fragments = []
        for frag in self.fragments:
            if self.player_pos.distance_to(frag) < self.PLAYER_RADIUS + self.FRAGMENT_RADIUS:
                collected_fragments.append(frag)
                self.score += 5
                reward += 5
                self._create_particles(frag, self.COLOR_FRAGMENT, 20)
                # sfx: fragment_collect.wav
        self.fragments = [f for f in self.fragments if f not in collected_fragments]

        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = False
        
        # 1. Caught by beast
        for beast in self.shadow_beasts:
            if self.player_pos.distance_to(beast['pos']) < self.PLAYER_RADIUS + self.BEAST_RADIUS:
                reward = -50
                terminated = True
                self.game_over_flag = True
                self._create_particles(self.player_pos, self.COLOR_BEAST, 50)
                # sfx: game_over.wav
                break
        
        # 2. All fragments collected (Win)
        if not terminated and not self.fragments:
            reward = 50
            terminated = True
            self.level += 1 # Progress to next level
            # sfx: level_complete.wav
        
        # 3. Max steps reached
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over_flag = True # Count as a loss
            reward = -25

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_valid_spawn_pos(self, existing_entities, min_x=0):
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(min_x + self.MIN_SPAWN_DISTANCE, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            )
            if all(pos.distance_to(e) > self.MIN_SPAWN_DISTANCE for e in existing_entities):
                return pos

    def _get_dist_to_nearest(self, pos, entity_list):
        if not entity_list:
            return float('inf')
        return min(pos.distance_to(e) for e in entity_list)

    def _handle_matching(self):
        nearest_body = None
        min_dist = float('inf')
        
        for body in self.celestial_bodies:
            if not body['matched']:
                dist = self.player_pos.distance_to(body['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_body = body
        
        if nearest_body and min_dist < self.TOUCH_RADIUS:
            # sfx: touch_celestial.wav
            if self.selected_body_idx is None:
                self.selected_body_idx = nearest_body['id']
                return 0.1 # Small reward for selecting
            else:
                selected = self.celestial_bodies[self.selected_body_idx]
                if nearest_body['id'] == selected['pair_id']:
                    # Successful Match
                    selected['matched'] = True
                    nearest_body['matched'] = True
                    self.selected_body_idx = None
                    self.score += 2
                    self._create_particles(selected['pos'], self.COLOR_CONSTELLATION, 15)
                    self._create_particles(nearest_body['pos'], self.COLOR_CONSTELLATION, 15)
                    # sfx: match_success.wav
                    return 2
                elif nearest_body['id'] == self.selected_body_idx:
                     # Deselect by touching same one again
                    self.selected_body_idx = None
                    return -0.1
                else:
                    # Wrong pair, deselect
                    self.selected_body_idx = None
                    # sfx: match_fail.wav
                    return -0.5
        return 0

    def _update_shadow_beasts(self):
        for beast in self.shadow_beasts:
            path = beast['path']
            target_pos = path[beast['target_idx']]
            
            if beast['pos'].distance_to(target_pos) < beast['speed']:
                beast['target_idx'] = (beast['target_idx'] + 1) % len(path)
            
            direction = (target_pos - beast['pos']).normalize()
            beast['pos'] += direction * beast['speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-1.5, 1.5)),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

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
            "level": self.level,
            "fragments_left": len(self.fragments),
        }

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength=3):
        center_int = (int(center.x), int(center.y))
        for i in range(glow_strength, 0, -1):
            alpha = int(100 / (i**1.5))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius + i * 2), glow_color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    def _render_game(self):
        # Starfield
        for x, y, size in self.starfield:
            pygame.draw.rect(self.screen, (100, 100, 120), (x, y, size, size))

        # Activated Constellations
        for body in self.celestial_bodies:
            if body['matched']:
                pair = self.celestial_bodies[body['pair_id']]
                if body['id'] < pair['id']: # Draw each line only once
                    p1 = (int(body['pos'].x), int(body['pos'].y))
                    p2 = (int(pair['pos'].x), int(pair['pos'].y))
                    pygame.draw.aaline(self.screen, self.COLOR_CONSTELLATION, p1, p2, 1)

        # Celestial Bodies
        for body in self.celestial_bodies:
            pos_int = (int(body['pos'].x), int(body['pos'].y))
            if body['matched']:
                color = self.COLOR_CELESTIAL_BODY_MATCHED
            elif self.selected_body_idx == body['id']:
                color = self.COLOR_CELESTIAL_BODY_SELECTED
                # Pulse effect for selected body
                pulse = abs(math.sin(self.steps * 0.2))
                self._draw_glow_circle(self.screen, color, body['pos'], self.CELESTIAL_RADIUS + pulse * 3, 2)
            else:
                color = self.COLOR_CELESTIAL_BODY_INACTIVE
            
            self.CELESTIAL_SYMBOLS[body['type']](self.screen, pos_int, color, self.CELESTIAL_RADIUS)

        # Starlight Fragments
        for frag in self.fragments:
            pulse = self.FRAGMENT_RADIUS + abs(math.sin(self.steps * 0.1 + frag.x)) * 2
            self._draw_glow_circle(self.screen, self.COLOR_FRAGMENT_GLOW, frag, pulse, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(frag.x), int(frag.y), int(self.FRAGMENT_RADIUS), self.COLOR_FRAGMENT)

        # Shadow Beasts
        for beast in self.shadow_beasts:
            self._draw_glow_circle(self.screen, self.COLOR_BEAST_GLOW, beast['pos'], self.BEAST_RADIUS, 4)
            # Draw spiky shape
            num_spikes = 7
            for i in range(num_spikes):
                angle = 2 * math.pi * i / num_spikes + self.steps * 0.05
                outer_r = self.BEAST_RADIUS + abs(math.sin(angle * 3 + self.steps * 0.1)) * 4
                p1 = beast['pos'] + pygame.Vector2(self.BEAST_RADIUS, 0).rotate_rad(angle)
                p2 = beast['pos'] + pygame.Vector2(outer_r, 0).rotate_rad(angle)
                pygame.draw.line(self.screen, self.COLOR_BEAST, p1, p2, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(beast['pos'].x), int(beast['pos'].y), int(self.BEAST_RADIUS*0.8), self.COLOR_BEAST)

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                p_color = (*p['color'], int(255 * (p['lifespan'] / 30.0)))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p_color)

        # Player
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER_GLOW, self.player_pos, self.PLAYER_RADIUS, 5)
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_RADIUS, 0)
        
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        fragments_text = self.font.render(f"FRAGMENTS: {len(self.fragments)}", True, self.COLOR_TEXT)
        self.screen.blit(fragments_text, (self.SCREEN_WIDTH - fragments_text.get_width() - 10, 10))
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be run by the autograder
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override render mode for human play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cosmic Stealth")
    clock = pygame.time.Clock()

    while not done:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Steps: {info['steps']}")
            # In a real game, you might wait here before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()