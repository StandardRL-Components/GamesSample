
# Generated: 2025-08-28T04:23:19.868846
# Source Brief: brief_05228.md
# Brief Index: 5228

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character. Get behind a block and press Space to push it. "
        "Push all colored blocks onto their matching goals before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down puzzle game. Maneuver your character to push blocks onto their goals against the clock. "
        "Each level increases the challenge with more complex layouts."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_SIZE = 40
    FPS = 30

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_WALL = (10, 10, 15)
    COLOR_WALL_TOP = (60, 60, 70)
    
    PLAYER_COLOR = (50, 255, 50)
    PLAYER_GLOW = (150, 255, 150)

    BLOCK_COLORS = {
        'r': (220, 50, 50),
        'b': (50, 100, 220),
        'y': (220, 220, 50),
    }
    BLOCK_HIGHLIGHTS = {
        'r': (255, 120, 120),
        'b': (120, 170, 255),
        'y': (255, 255, 120),
    }
    GOAL_COLORS = {
        'R': (80, 40, 40),
        'B': (40, 60, 80),
        'Y': (80, 80, 40),
    }
    GOAL_SYMBOLS = {
        'R': (220, 50, 50),
        'B': (50, 100, 220),
        'Y': (220, 220, 50),
    }

    LEVEL_DEFINITIONS = [
        [
            "WWWWWWWWWWWWWWWW",
            "W P            W",
            "W     r   b    W",
            "W              W",
            "W         y    W",
            "W              W",
            "W   R   B   Y  W",
            "W              W",
            "W              W",
            "WWWWWWWWWWWWWWWW"
        ],
        [
            "WWWWWWWWWWWWWWWW",
            "W WWWWWWWWWWWW W",
            "W W r      b W W",
            "W W WW WWWWW W W",
            "W W  P W     W W",
            "W WWWW W y R W W",
            "W      W   W W W",
            "W B      Y W W W",
            "W WWWWWWWWWWWW W",
            "WWWWWWWWWWWWWWWW"
        ],
        [
            "WWWWWWWWWWWWWWWW",
            "W     P      y W",
            "W b WWW WWWWWW W",
            "W   W r      W W",
            "W   W WWWWWW W W",
            "W r W      r W W",
            "W   WWWWWW   W W",
            "W B        Y W W",
            "W     R R    W W",
            "WWWWWWWWWWWWWWWW"
        ]
    ]

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Etc...
        self.render_mode = render_mode
        self.np_random = None

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        self.time_remaining = 0.0
        self.player_pos = (0, 0)
        self.player_visual_pos = (0, 0)
        self.player_facing = (0, 1) # (dx, dy)
        self.walls = set()
        self.blocks = []
        self.goals = defaultdict(list)
        self.particles = []
        self.move_cooldown = 0
        self.push_cooldown = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        
        self._load_level(self.current_level)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _load_level(self, level_num):
        if level_num > len(self.LEVEL_DEFINITIONS):
            self.game_over = True
            return

        self.time_remaining = 60.0
        self.walls = set()
        self.blocks = []
        self.goals = defaultdict(list)
        self.particles = []
        
        level_data = self.LEVEL_DEFINITIONS[level_num - 1]
        block_id_counter = 0

        for y, row in enumerate(level_data):
            for x, char in enumerate(row):
                pos = (x, y)
                visual_pos = (x * self.TILE_SIZE, y * self.TILE_SIZE)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'P':
                    self.player_pos = pos
                    self.player_visual_pos = visual_pos
                    self.player_facing = (0, 1)
                elif char in self.BLOCK_COLORS:
                    self.blocks.append({
                        'id': block_id_counter,
                        'pos': pos,
                        'visual_pos': visual_pos,
                        'color_char': char,
                        'on_goal': False
                    })
                    block_id_counter += 1
                elif char in self.GOAL_COLORS:
                    self.goals[char.lower()].append(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.time_remaining -= 1.0 / self.FPS
        self.steps += 1

        reward = -0.01 # Small penalty for time passing, encourages efficiency

        # Update cooldowns
        if self.move_cooldown > 0: self.move_cooldown -= 1
        if self.push_cooldown > 0: self.push_cooldown -= 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Handle Movement ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if movement in move_map:
            self.player_facing = move_map[movement]
            if self.move_cooldown == 0:
                next_pos = (self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1])
                occupied_by_block = any(b['pos'] == next_pos for b in self.blocks)
                if next_pos not in self.walls and not occupied_by_block:
                    self.player_pos = next_pos
                    self.move_cooldown = 4 # 4 frames cooldown
        
        # --- Handle Push ---
        if space_held and self.push_cooldown == 0:
            pos_in_front = (self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1])
            block_to_push = next((b for b in self.blocks if b['pos'] == pos_in_front), None)
            
            if block_to_push:
                pos_behind_block = (pos_in_front[0] + self.player_facing[0], pos_in_front[1] + self.player_facing[1])
                is_clear = pos_behind_block not in self.walls and not any(b['pos'] == pos_behind_block for b in self.blocks)
                
                if is_clear:
                    # Calculate reward before moving
                    old_on_goal = self._is_on_goal(block_to_push['pos'], block_to_push['color_char'])
                    new_on_goal = self._is_on_goal(pos_behind_block, block_to_push['color_char'])

                    if not old_on_goal and new_on_goal:
                        reward += 1.0 # Moved onto correct goal
                    elif old_on_goal and not new_on_goal:
                        reward -= 0.5 # Moved off a goal

                    # Move block
                    block_to_push['pos'] = pos_behind_block
                    self.push_cooldown = 10 # 10 frames cooldown
                    # Add particles for feedback
                    self._create_particles(block_to_push['visual_pos'], block_to_push['color_char'])
                    # Sound placeholder: # pygame.mixer.Sound('push.wav').play()

        # --- Update Visuals & Game State ---
        self._update_visual_positions()
        self._update_particles()
        
        # Check if blocks are on goals
        for block in self.blocks:
            block['on_goal'] = self._is_on_goal(block['pos'], block['color_char'])

        # Check for level completion
        if all(b['on_goal'] for b in self.blocks):
            reward += 100
            self.score += 1000 + int(self.time_remaining * 10) # Time bonus
            self.current_level += 1
            if self.current_level > len(self.LEVEL_DEFINITIONS):
                self.game_over = True # Game won
            else:
                self._load_level(self.current_level)
                # Sound placeholder: # pygame.mixer.Sound('level_complete.wav').play()

        # Check for time out
        terminated = self.game_over or self.time_remaining <= 0
        if self.time_remaining <= 0 and not self.game_over:
            reward -= 100
            self.game_over = True
            # Sound placeholder: # pygame.mixer.Sound('game_over.wav').play()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _is_on_goal(self, pos, color_char):
        return pos in self.goals.get(color_char, [])

    def _update_visual_positions(self):
        # Interpolate player
        target_x, target_y = self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE
        self.player_visual_pos = (
            self.player_visual_pos[0] * 0.75 + target_x * 0.25,
            self.player_visual_pos[1] * 0.75 + target_y * 0.25
        )
        # Interpolate blocks
        for block in self.blocks:
            target_x, target_y = block['pos'][0] * self.TILE_SIZE, block['pos'][1] * self.TILE_SIZE
            block['visual_pos'] = (
                block['visual_pos'][0] * 0.7 + target_x * 0.3,
                block['visual_pos'][1] * 0.7 + target_y * 0.3
            )

    def _create_particles(self, pos, color_char):
        color = self.BLOCK_COLORS[color_char]
        for _ in range(15):
            self.particles.append({
                'pos': [pos[0] + self.TILE_SIZE / 2, pos[1] + self.TILE_SIZE / 2],
                'vel': [random.uniform(-3, 3), random.uniform(-3, 3)],
                'life': random.randint(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw goals
        for color_char, positions in self.goals.items():
            goal_color = self.GOAL_COLORS[color_char.upper()]
            symbol_color = self.GOAL_SYMBOLS[color_char.upper()]
            for pos in positions:
                rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, goal_color, rect)
                pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.TILE_SIZE // 4, symbol_color)

        # Draw walls
        for x, y in self.walls:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.line(self.screen, self.COLOR_WALL_TOP, rect.topleft, rect.topright, 2)

        # Draw blocks
        for block in self.blocks:
            vx, vy = block['visual_pos']
            rect = pygame.Rect(vx + 4, vy + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            color = self.BLOCK_COLORS[block['color_char']]
            highlight = self.BLOCK_HIGHLIGHTS[block['color_char']]
            
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.line(self.screen, highlight, (rect.left + 2, rect.top + 2), (rect.right - 2, rect.top + 2), 2)

            if block['on_goal']:
                pygame.gfxdraw.filled_circle(self.screen, int(vx + self.TILE_SIZE / 2), int(vy + self.TILE_SIZE / 2), 6, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, int(vx + self.TILE_SIZE / 2), int(vy + self.TILE_SIZE / 2), 6, (255, 255, 255))

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], 3, 3))

        # Draw player
        px, py = self.player_visual_pos[0] + self.TILE_SIZE / 2, self.player_visual_pos[1] + self.TILE_SIZE / 2
        size = self.TILE_SIZE / 3
        angle = math.atan2(self.player_facing[1], self.player_facing[0]) - math.pi / 2
        
        points = [
            (px + size * math.sin(angle), py - size * math.cos(angle)),
            (px + (size/2) * math.sin(angle + 2.5), py - (size/2) * math.cos(angle + 2.5)),
            (px + (size/2) * math.sin(angle - 2.5), py - (size/2) * math.cos(angle - 2.5)),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.PLAYER_COLOR)
        pygame.gfxdraw.aapolygon(self.screen, points, self.PLAYER_GLOW)

    def _render_ui(self):
        # Render Level
        level_text = self.font_small.render(f"Level: {self.current_level}", True, (200, 200, 255))
        self.screen.blit(level_text, (10, 5))

        # Render Timer
        time_color = (200, 200, 255)
        if self.time_remaining < 10:
            time_color = (255, 100, 100)
        time_text = self.font_small.render(f"Time: {max(0, self.time_remaining):.1f}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 5))

        # Render Score
        score_text = self.font_small.render(f"Score: {self.score}", True, (200, 200, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - score_text.get_height() - 5))

        # Render Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.current_level > len(self.LEVEL_DEFINITIONS):
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2 - 20))
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, (220, 220, 220))
            self.screen.blit(final_score_text, (self.SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, self.SCREEN_HEIGHT // 2 + 30))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for manual play
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running and not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)
            running = False
            
    env.close()