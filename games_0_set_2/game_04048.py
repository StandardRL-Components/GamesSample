
# Generated: 2025-08-28T01:16:53.674294
# Source Brief: brief_04048.md
# Brief Index: 4048

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (Up, Down, Left, Right) to move your red avatar and push the colored blocks onto their matching goals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style puzzle game where you must push blocks to their goals within a limited number of moves. Plan your pushes carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    TILE_SIZE = 40
    MAX_STEPS = 1000
    ANIMATION_DURATION = 6 # Frames for animation

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_WALL = (80, 80, 90)
    COLOR_WALL_SHADOW = (60, 60, 70)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_GOAL = (50, 200, 50, 100) # RGBA for transparency
    
    BLOCK_COLORS = [
        ((66, 135, 245), (40, 100, 200)), # Blue
        ((245, 188, 66), (200, 150, 40)), # Orange
        ((212, 66, 245), (170, 40, 200)), # Purple
        ((245, 242, 66), (200, 190, 40)), # Yellow
    ]

    LEVELS = [
        {
            "moves": 50,
            "layout": [
                "################",
                "#              #",
                "#  P      1A   #",
                "#              #",
                "#    2B        #",
                "#              #",
                "#              #",
                "#              #",
                "#              #",
                "################",
            ]
        },
        {
            "moves": 45,
            "layout": [
                "################",
                "# P            #",
                "# #######  1A  #",
                "# #     #      #",
                "# #  2  #      #",
                "# #  B  #      #",
                "# #     #      #",
                "# #######      #",
                "#              #",
                "################",
            ]
        },
        {
            "moves": 40,
            "layout": [
                "################",
                "#P  1#         #",
                "#    #    2    #",
                "#  A #    B    #",
                "#    #         #",
                "# 3C #         #",
                "#    #         #",
                "#    #         #",
                "#              #",
                "################",
            ]
        }
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        
        # Etc...        
        self.current_level = 0
        self.total_score = 0
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _load_level(self, level_index):
        if level_index >= len(self.LEVELS):
            self.game_over = True
            self.win_condition = True
            return

        level_data = self.LEVELS[level_index]
        self.moves_remaining = level_data["moves"]
        layout = level_data["layout"]

        self.walls = []
        self.blocks = []
        self.goals = {} # Using a dict for easy goal lookup by block_id
        
        block_counter = 0
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                pos = [c, r]
                if char == '#':
                    self.walls.append(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char.isdigit():
                    block_id = int(char)
                    color_pair = self.BLOCK_COLORS[block_counter % len(self.BLOCK_COLORS)]
                    self.blocks.append({'id': block_id, 'pos': pos, 'color': color_pair[0], 'shadow': color_pair[1]})
                    block_counter += 1
                elif char.isalpha() and char.isupper():
                    goal_id = ord(char) - ord('A') + 1
                    self.goals[goal_id] = pos
        
        self.win_condition = False
        self.animations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        
        self._load_level(self.current_level)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean (unused)
        shift_held = action[2] == 1  # Boolean (unused)
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Advance animations if any are active, this allows smooth visuals
        # even with auto_advance=False. The agent must send no-ops to see animations.
        if self.animations:
            self.steps += 1
            reward = 0
            terminated = False
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Handle no-op
        if movement == 0:
            self.steps += 1
            return self._get_observation(), 0, False, False, self._get_info()

        # Update game logic for a move action
        self.steps += 1
        self.moves_remaining -= 1
        reward = 0.0

        old_block_dists = {b['id']: self._dist_to_goal(b['pos'], b['id']) for b in self.blocks}
        
        moved_blocks = self._handle_player_move(movement)
        
        reward += self._calculate_reward(moved_blocks, old_block_dists)
        
        terminated, win_reward, loss_penalty = self._check_termination()

        reward += win_reward + loss_penalty
        self.score = self.total_score # Sync score for info dict
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_move(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}
        dx, dy = direction_map[movement]

        player_next_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
        
        if player_next_pos in self.walls:
            # sfx: bump_wall.wav
            return []

        blocks_to_push = []
        check_pos = list(player_next_pos)
        while True:
            block_at_pos = self._get_block_at(check_pos)
            if block_at_pos:
                blocks_to_push.append(block_at_pos)
                check_pos[0] += dx
                check_pos[1] += dy
            else:
                break
        
        if blocks_to_push:
            last_block_next_pos = [blocks_to_push[-1]['pos'][0] + dx, blocks_to_push[-1]['pos'][1] + dy]
            if last_block_next_pos in self.walls or self._get_block_at(last_block_next_pos):
                # sfx: bump_block.wav
                return []

        # sfx: push.wav
        old_player_pos = list(self.player_pos)
        self.player_pos = player_next_pos
        self._add_animation(self.player_pos, old_player_pos, 'player')

        for block in reversed(blocks_to_push):
            old_block_pos = list(block['pos'])
            block['pos'][0] += dx
            block['pos'][1] += dy
            self._add_animation(block['pos'], old_block_pos, 'block', block)
        
        return blocks_to_push

    def _calculate_reward(self, moved_blocks, old_block_dists):
        reward = 0.0
        for block in moved_blocks:
            new_dist = self._dist_to_goal(block['pos'], block['id'])
            old_dist = old_block_dists[block['id']]
            
            if new_dist < old_dist: reward += 0.1
            elif new_dist > old_dist: reward -= 0.1
            
            if new_dist == 0:
                reward += 5
                # sfx: block_on_goal.wav

            if self._is_stuck_in_corner(block['pos']):
                reward -= 1
        return reward

    def _check_termination(self):
        win_reward = 0
        loss_penalty = 0
        
        level_complete = all(self._dist_to_goal(b['pos'], b['id']) == 0 for b in self.blocks)
        
        if level_complete:
            win_reward += 100
            self.total_score += 100 + self.moves_remaining
            self.current_level += 1
            if self.current_level >= len(self.LEVELS):
                self.game_over = True
                self.win_condition = True
            else:
                self._load_level(self.current_level)
                # sfx: level_complete.wav
        
        loss_condition = self.moves_remaining <= 0
        timeout = self.steps >= self.MAX_STEPS
        terminated = level_complete or loss_condition or timeout

        if loss_condition and not level_complete:
            loss_penalty -= 50
            self.game_over = True
            # sfx: game_over.wav

        if timeout:
            self.game_over = True
        
        return terminated, win_reward, loss_penalty
    
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

    def _render_game(self):
        # Draw grid, goals, walls
        self._draw_static_scenery()
        
        static_blocks = list(self.blocks)
        static_player = True

        active_animations = []
        for anim in self.animations:
            anim['progress'] += 1.0 / self.ANIMATION_DURATION
            if anim['progress'] >= 1.0: anim['progress'] = 1.0
            else: active_animations.append(anim)

            p = 1 - (1 - anim['progress'])**3 # Ease-out curve
            
            start_px = [self.grid_offset_x + anim['start_pos'][c] * self.TILE_SIZE for c in range(2)]
            end_px = [self.grid_offset_x + anim['end_pos'][c] * self.TILE_SIZE for c in range(2)]
            draw_pos = [start_px[c] + (end_px[c] - start_px[c]) * p for c in range(2)]
            
            if anim['type'] == 'player':
                static_player = False
                self._draw_player(*draw_pos)
            elif anim['type'] == 'block':
                if anim['obj_ref'] in static_blocks: static_blocks.remove(anim['obj_ref'])
                self._draw_block(*draw_pos, anim['obj_ref']['color'], anim['obj_ref']['shadow'])
        
        self.animations = active_animations

        if static_player:
            pos = [self.grid_offset_x + self.player_pos[c] * self.TILE_SIZE for c in range(2)]
            self._draw_player(*pos)
        
        for block in static_blocks:
            pos = [self.grid_offset_x + block['pos'][c] * self.TILE_SIZE for c in range(2)]
            self._draw_block(*pos, block['color'], block['shadow'])

    def _draw_static_scenery(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.GRID_HEIGHT * self.TILE_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.GRID_WIDTH * self.TILE_SIZE, y))

        for block_id, pos in self.goals.items():
            rect = pygame.Rect(self.grid_offset_x + pos[0] * self.TILE_SIZE, self.grid_offset_y + pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            block_color = self.BLOCK_COLORS[(block_id-1) % len(self.BLOCK_COLORS)][0]
            goal_color = (*block_color, 60)
            pygame.draw.rect(surface, goal_color, surface.get_rect(), border_radius=8)
            self.screen.blit(surface, rect.topleft)

        for pos in self.walls:
            rect = pygame.Rect(self.grid_offset_x + pos[0] * self.TILE_SIZE, self.grid_offset_y + pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL_SHADOW, rect.move(0, 4))
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _draw_player(self, x, y):
        rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE).inflate(-10, -10)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=6)

    def _draw_block(self, x, y, color, shadow_color):
        rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE).inflate(-4, -4)
        shadow_rect = rect.move(0, 3)
        pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=8)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)

    def _render_ui(self):
        moves_text = self.font_large.render(f"{self.moves_remaining}", True, (255, 255, 255))
        moves_label = self.font_small.render("MOVES", True, (150, 150, 180))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - 80, 20))
        self.screen.blit(moves_label, (self.SCREEN_WIDTH - 80, 60))

        score_text = self.font_small.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        level_text = self.font_small.render(f"LEVEL: {self.current_level + 1}", True, (255, 255, 255))
        self.screen.blit(level_text, (20, 45))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition else "GAME OVER"
            msg_render = self.font_large.render(message, True, (255, 255, 255))
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_render, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def _get_block_at(self, pos):
        for block in self.blocks:
            if block['pos'] == pos: return block
        return None

    def _is_stuck_in_corner(self, pos):
        x, y = pos
        is_wall = lambda p: p in self.walls
        if (is_wall([x, y-1]) and is_wall([x-1, y])) or \
           (is_wall([x, y-1]) and is_wall([x+1, y])) or \
           (is_wall([x, y+1]) and is_wall([x-1, y])) or \
           (is_wall([x, y+1]) and is_wall([x+1, y])):
            return True
        return False
        
    def _dist_to_goal(self, pos, block_id):
        goal_pos = self.goals.get(block_id)
        if not goal_pos: return float('inf')
        return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])

    def _add_animation(self, new_pos, old_pos, anim_type, obj_ref=None):
        self.animations.append({
            'start_pos': old_pos, 'end_pos': new_pos, 'progress': 0.0,
            'type': anim_type, 'obj_ref': obj_ref,
        })

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if action[0] != 0 or env.animations:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if terminated:
                print("--- Episode Finished ---")
                print(f"Final Info: {info}")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()