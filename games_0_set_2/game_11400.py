import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:55:23.223893
# Source Brief: brief_01400.md
# Brief Index: 1400
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
    A turn-based strategy game where players control Bub and Bob to trap enemies in bubbles.
    The game is adapted to a real-time action space where each step processes one player
    action (move, bubble, or switch) and then executes the corresponding enemy turn.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control Bub and Bob to trap enemies in bubbles. Clear all enemies from the screen to win, but don't let them touch you!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move, space to shoot a bubble, and shift to switch characters."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 32
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = (self.HEIGHT - 80) // self.GRID_SIZE  # Reserve 80px for UI
        self.UI_HEIGHT = 80
        self.MAX_STEPS = 2000
        self.BUBBLE_LIFETIME = 8  # in game turns
        self.PARTICLE_LIFETIME = 20 # in frames

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_BUB = (60, 160, 255)
        self.COLOR_BUB_GLOW = (120, 200, 255)
        self.COLOR_BOB = (100, 220, 100)
        self.COLOR_BOB_GLOW = (160, 255, 160)
        self.COLOR_ENEMY_1 = (255, 100, 100)
        self.COLOR_ENEMY_2 = (255, 150, 80)
        self.COLOR_ENEMY_3 = (200, 100, 255)
        self.ENEMY_COLORS = [self.COLOR_ENEMY_1, self.COLOR_ENEMY_2, self.COLOR_ENEMY_3]
        self.COLOR_BUBBLE = (230, 230, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_m = pygame.font.Font(None, 28)
            self.font_l = pygame.font.Font(None, 40)
        except:
            self.font_m = pygame.font.SysFont('sans', 28)
            self.font_l = pygame.font.SysFont('sans', 40)


        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_bub = {}
        self.player_bob = {}
        self.active_player_name = 'bub'
        self.enemies = []
        self.bubbles = []
        self.particles = []
        self.enemy_types_unlocked = 1
        self.enemy_base_speed = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_bub = {'pos': np.array([3, self.GRID_H - 2]), 'alive': True}
        self.player_bob = {'pos': np.array([self.GRID_W - 4, self.GRID_H - 2]), 'alive': True}
        self.active_player_name = 'bub'

        self.enemies = []
        self.bubbles = []
        self.particles = []

        self.enemy_types_unlocked = 1
        self.enemy_base_speed = 0.5
        self._spawn_initial_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Player Action Phase ---
        player_acted = self._handle_player_action(action)
        
        # --- Enemy & World Phase (only if player acted) ---
        if player_acted:
            self.steps += 1
            bubble_pop_reward, enemies_killed = self._update_bubbles()
            reward += bubble_pop_reward
            
            enemy_move_reward, players_popped = self._update_enemies()
            reward += enemy_move_reward
            
            # Apply penalties for player loss
            if players_popped > 0:
                reward -= 10.0 * players_popped # Sfx: player_pop.wav

            # Difficulty scaling
            self._update_difficulty()
        elif action[0] == 0 and action[1] == 0 and action[2] == 0:
            # Small penalty for no-op, since auto_advance is False
            reward -= 0.01

        # --- Termination Check ---
        if not self.enemies:
            reward += 100.0 # Sfx: level_clear.wav
            self.game_over = True
        elif not self.player_bub['alive'] and not self.player_bob['alive']:
            self.game_over = True # Sfx: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info(),
        )

    def _handle_player_action(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        player_acted = False

        active_player = self.player_bub if self.active_player_name == 'bub' else self.player_bob
        if not active_player['alive']:
            # If active player is dead, any action (except no-op) switches to the other player if alive
            if (movement != 0 or space_pressed or shift_pressed):
                 if (self.active_player_name == 'bub' and self.player_bob['alive']) or \
                    (self.active_player_name == 'bob' and self.player_bub['alive']):
                    self._switch_player()
                    return True # Switching is an action
            return False

        if shift_pressed:
            # Sfx: switch_char.wav
            self._switch_player()
            player_acted = True
        elif space_pressed:
            # Sfx: bubble_create.wav
            self._place_bubble(active_player['pos'])
            player_acted = True
        elif movement != 0:
            # Sfx: player_move.wav
            self._move_player(active_player, movement)
            player_acted = True
            
        return player_acted

    def _switch_player(self):
        if self.active_player_name == 'bub' and self.player_bob['alive']:
            self.active_player_name = 'bob'
        elif self.active_player_name == 'bob' and self.player_bub['alive']:
            self.active_player_name = 'bub'

    def _place_bubble(self, pos):
        # Prevent placing bubbles on top of each other
        if any(np.array_equal(b['pos'], pos) for b in self.bubbles):
            return
        
        bubble = {
            'pos': pos.copy(),
            'timer': self.BUBBLE_LIFETIME,
            'creation_tick': self.steps,
            'trapped_enemy': None
        }
        self.bubbles.append(bubble)

    def _move_player(self, player, movement):
        old_pos = player['pos'].copy()
        delta = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement, [0, 0])
        new_pos = player['pos'] + np.array(delta)

        other_player = self.player_bob if player is self.player_bub else self.player_bub
        
        # Check boundaries and collision with other player
        if (0 <= new_pos[0] < self.GRID_W and
            0 <= new_pos[1] < self.GRID_H and
            (not other_player['alive'] or not np.array_equal(new_pos, other_player['pos']))):
            player['pos'] = new_pos
    
    def _update_bubbles(self):
        reward = 0
        enemies_killed_this_turn = 0
        
        for bubble in self.bubbles[:]:
            bubble['timer'] -= 1
            if bubble['timer'] <= 0:
                # Sfx: bubble_pop.wav
                if bubble['trapped_enemy']:
                    bubble['trapped_enemy']['alive'] = False
                    reward += 1.0 # Reward for popping a trapped enemy
                    enemies_killed_this_turn += 1
                self._create_particles(self._grid_to_pixel(bubble['pos']), self.COLOR_BUBBLE)
                self.bubbles.remove(bubble)
        
        # Remove killed enemies from the main list
        self.enemies = [e for e in self.enemies if e.get('alive', True)]
        return reward, enemies_killed_this_turn

    def _update_enemies(self):
        reward = 0
        players_popped = 0
        
        for enemy in self.enemies:
            if enemy['trapped_in']:
                # If bubble popped this turn, the reference might be stale
                if enemy['trapped_in'] not in self.bubbles:
                    enemy['trapped_in'] = None
                continue

            # --- Enemy Movement ---
            enemy['move_progress'] += enemy['speed']
            moves_to_make = int(enemy['move_progress'])
            if moves_to_make > 0:
                enemy['move_progress'] -= moves_to_make
            
            for _ in range(moves_to_make):
                # Find closest alive player
                target = self._find_closest_player(enemy['pos'])
                if not target: continue
                
                # Simple pathfinding
                dx, dy = target['pos'] - enemy['pos']
                new_pos = enemy['pos'].copy()
                if abs(dx) > abs(dy):
                    new_pos[0] += np.sign(dx)
                elif dy != 0:
                    new_pos[1] += np.sign(dy)
                
                if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                    enemy['pos'] = new_pos


            # --- Collision Checks ---
            # Check for trapping in bubbles
            for bubble in self.bubbles:
                if not bubble['trapped_enemy'] and np.array_equal(enemy['pos'], bubble['pos']):
                    enemy['trapped_in'] = bubble
                    bubble['trapped_enemy'] = enemy
                    reward += 0.1 # Reward for trapping
                    # Sfx: enemy_trapped.wav
                    break
            
            # Check for popping players
            for player_name, player in [('bub', self.player_bub), ('bob', self.player_bob)]:
                if player['alive'] and np.array_equal(enemy['pos'], player['pos']):
                    player['alive'] = False
                    players_popped += 1
                    self._create_particles(self._grid_to_pixel(player['pos']), self.COLOR_BUB if player_name == 'bub' else self.COLOR_BOB)

        return reward, players_popped

    def _find_closest_player(self, pos):
        alive_players = []
        if self.player_bub['alive']: alive_players.append(self.player_bub)
        if self.player_bob['alive']: alive_players.append(self.player_bob)
        if not alive_players: return None

        distances = [np.sum(np.abs(p['pos'] - pos)) for p in alive_players]
        return alive_players[np.argmin(distances)]

    def _update_difficulty(self):
        # Increase speed
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_base_speed = min(2.0, self.enemy_base_speed + 0.1)
            for e in self.enemies:
                 e['speed'] = self.enemy_base_speed

        # Unlock new enemy types
        if self.steps > 0 and self.steps % 500 == 0 and self.enemy_types_unlocked < 3:
            self.enemy_types_unlocked += 1
            self._spawn_new_enemy()
    
    def _spawn_initial_enemies(self):
        self._spawn_new_enemy()

    def _spawn_new_enemy(self):
        spawn_pos = np.array([self.np_random.integers(0, self.GRID_W), 0])
        enemy_type = self.np_random.integers(0, self.enemy_types_unlocked)
        
        enemy = {
            'pos': spawn_pos,
            'type': enemy_type,
            'speed': self.enemy_base_speed,
            'move_progress': 0.0,
            'trapped_in': None,
            'alive': True
        }
        self.enemies.append(enemy)

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
            "enemies_left": len(self.enemies),
            "bub_alive": self.player_bub['alive'],
            "bob_alive": self.player_bob['alive'],
        }

    def _grid_to_pixel(self, grid_pos):
        return (
            int(grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2 + self.UI_HEIGHT)
        )
    
    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = x * self.GRID_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.UI_HEIGHT), (px, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.GRID_SIZE + self.UI_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        self._update_and_draw_particles()
        self._draw_bubbles()
        self._draw_enemies()
        if self.player_bub['alive']: self._draw_character(self.player_bub, self.COLOR_BUB, self.COLOR_BUB_GLOW, self.active_player_name == 'bub')
        if self.player_bob['alive']: self._draw_character(self.player_bob, self.COLOR_BOB, self.COLOR_BOB_GLOW, self.active_player_name == 'bob')

    def _draw_character(self, player, color, glow_color, is_active):
        pos_px = self._grid_to_pixel(player['pos'])
        radius = self.GRID_SIZE // 2 - 2

        if is_active:
            glow_radius = radius + 5 + 3 * math.sin(pygame.time.get_ticks() * 0.005)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], int(glow_radius), (*glow_color, 60))
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], int(glow_radius), (*glow_color, 100))

        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, color)

    def _draw_enemies(self):
        for enemy in self.enemies:
            pos_px = self._grid_to_pixel(enemy['pos'])
            radius = self.GRID_SIZE // 2 - 4
            color = self.ENEMY_COLORS[enemy['type']]

            if enemy['trapped_in']:
                # Jiggle when trapped
                pos_px = (pos_px[0] + self.np_random.integers(-2, 3), pos_px[1] + self.np_random.integers(-2, 3))
            
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, color)

    def _draw_bubbles(self):
        for bubble in self.bubbles:
            pos_px = self._grid_to_pixel(bubble['pos'])
            
            # Expansion animation
            age = self.steps - bubble['creation_tick']
            max_radius = self.GRID_SIZE // 2
            radius = min(max_radius, int(max_radius * (age + 1) / 2.0))

            # Pulsing effect
            pulse = 1 + 0.08 * math.sin(pygame.time.get_ticks() * 0.002 + bubble['pos'][0])
            display_radius = int(radius * pulse)
            
            # Draw bubble
            alpha = 100 + int(50 * math.sin(pygame.time.get_ticks() * 0.005))
            if display_radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], display_radius, (*self.COLOR_BUBBLE, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], display_radius, (*self.COLOR_BUBBLE, alpha + 50))

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifetime': self.PARTICLE_LIFETIME,
                'color': color
            })
    
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            
            if p['lifetime'] <= 0:
                self.particles.remove(p)
                continue

            alpha = int(255 * (p['lifetime'] / self.PARTICLE_LIFETIME))
            radius = int(5 * (p['lifetime'] / self.PARTICLE_LIFETIME))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], alpha))

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (25, 25, 40), (0, 0, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT-1), (self.WIDTH, self.UI_HEIGHT-1))
        
        # Player Portraits
        bub_active = self.active_player_name == 'bub' and self.player_bub['alive']
        bob_active = self.active_player_name == 'bob' and self.player_bob['alive']
        
        # Bub
        self._render_portrait(40, 40, self.COLOR_BUB, self.COLOR_BUB_GLOW, self.player_bub['alive'], bub_active)
        self._render_text("BUB", self.font_m, self.COLOR_TEXT, (80, 25))

        # Bob
        self._render_portrait(self.WIDTH - 40, 40, self.COLOR_BOB, self.COLOR_BOB_GLOW, self.player_bob['alive'], bob_active)
        self._render_text("BOB", self.font_m, self.COLOR_TEXT, (self.WIDTH - 130, 25))

        # Score and Enemies
        score_str = f"SCORE: {int(self.score)}"
        enemies_str = f"ENEMIES: {len(self.enemies)}"
        
        self._render_text(score_str, self.font_l, self.COLOR_TEXT, (self.WIDTH/2 - self.font_l.size(score_str)[0]/2, 10))
        self._render_text(enemies_str, self.font_m, self.COLOR_TEXT, (self.WIDTH/2 - self.font_m.size(enemies_str)[0]/2, 50))


    def _render_portrait(self, x, y, color, glow_color, alive, active):
        radius = 25
        if not alive:
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (50, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (80, 80, 80))
            # Draw X
            pygame.draw.line(self.screen, (200, 50, 50), (x-10, y-10), (x+10, y+10), 4)
            pygame.draw.line(self.screen, (200, 50, 50), (x-10, y+10), (x+10, y-10), 4)
            return

        if active:
            glow_radius = radius + 4 + 2 * math.sin(pygame.time.get_ticks() * 0.005)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(glow_radius), (*glow_color, 60))
        
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Unset the dummy driver for local rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Bubble Trouble Strategy")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(env.user_guide)
    print("R:      Reset environment")
    print("----------------\n")

    while not done:
        # Action defaults
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Only step if an action is taken to feel more "turn-based"
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Enemies: {info['enemies_left']}")
            if terminated or truncated:
                print("Game Over!")
                obs, info = env.reset()
        else: # If no action, still need to get observation for rendering
            obs = env._get_observation()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control the speed of manual play

    env.close()