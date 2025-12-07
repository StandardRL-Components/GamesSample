import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    """
    An isometric puzzle game where the player must push crystals onto pressure plates
    to illuminate them all within a step limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move and change facing direction. Press Space to push a crystal."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a procedurally generated cavern, pushing crystals onto pressure plates to solve the puzzle before you run out of moves."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 14, 14
        self.NUM_CRYSTALS = 4
        self.MAX_STEPS = 300

        # Visuals
        self.TILE_WIDTH, self.TILE_HEIGHT = 60, 30
        self.TILE_W_HALF, self.TILE_H_HALF = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 60
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_TILE = (25, 35, 55)
        self.COLOR_TILE_EDGE = (40, 50, 75)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_WALL_TOP = (80, 90, 110)
        self.COLOR_PLAYER = (255, 255, 0)
        self.CRYSTAL_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 150, 255), (255, 80, 255)
        ]
        self.COLOR_CRYSTAL_INACTIVE = (100, 100, 100)
        self.COLOR_PLATE_INACTIVE = (45, 55, 80)
        self.COLOR_PLATE_ACTIVE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rng = None
        self.player_pos = (0, 0)
        self.player_facing_dir = (0, 1) # N, S, E, W -> (0,-1), (0,1), (1,0), (-1,0)
        self.walls = set()
        self.crystals = []
        self.plates = []
        self.particles = []

        # Initialize state by calling reset
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # If no seed is provided, create a new generator
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = -0.01  # Small penalty for each step

        prev_lit_crystals = sum(1 for c in self.crystals if c['is_lit'])
        prev_active_plates = sum(1 for p in self.plates if p['is_active'])

        # --- Action Logic ---
        if space_pressed:
            # PUSH action
            self._handle_push()
        else:
            # MOVE action
            self._handle_move(movement)
        
        # --- Update Game State ---
        self._update_plates_and_crystals()

        # --- Calculate Rewards ---
        new_lit_crystals = sum(1 for c in self.crystals if c['is_lit'])
        new_active_plates = sum(1 for p in self.plates if p['is_active'])

        if new_lit_crystals > prev_lit_crystals:
            reward += (new_lit_crystals - prev_lit_crystals) * 1.0
        if new_active_plates > prev_active_plates:
            reward += (new_active_plates - prev_active_plates) * 5.0

        self.steps += 1
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100
                self.score += 100
            else:
                reward -= 100
                self.score -= 100

        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_move(self, movement_action):
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        if movement_action not in direction_map:
            return # No-op

        direction = direction_map[movement_action]
        self.player_facing_dir = direction
        
        target_pos = (self.player_pos[0] + direction[0], self.player_pos[1] + direction[1])

        if self._is_pos_vacant(target_pos):
            self.player_pos = target_pos

    def _handle_push(self):
        direction = self.player_facing_dir
        crystal_pos = (self.player_pos[0] + direction[0], self.player_pos[1] + direction[1])
        
        crystal_to_push = None
        for c in self.crystals:
            if c['pos'] == crystal_pos:
                crystal_to_push = c
                break
        
        if crystal_to_push:
            destination_pos = (crystal_pos[0] + direction[0], crystal_pos[1] + direction[1])
            if self._is_pos_vacant(destination_pos):
                crystal_to_push['pos'] = destination_pos
                self.player_pos = crystal_pos # Player moves into the crystal's old spot
                # Sound: push_crystal.wav
                self._create_spark_particles(crystal_pos, crystal_to_push['color'])


    def _update_plates_and_crystals(self):
        # Deactivate all plates first
        for plate in self.plates:
            plate['is_active'] = False
        
        # Activate plates that have a crystal on them
        crystal_positions = {c['pos'] for c in self.crystals}
        for plate in self.plates:
            if plate['pos'] in crystal_positions:
                plate['is_active'] = True

        # Update crystal lit status based on their linked plate
        for crystal in self.crystals:
            was_lit = crystal['is_lit']
            crystal['is_lit'] = False
            for plate in self.plates:
                if plate['crystal_id'] == crystal['id'] and plate['is_active']:
                    crystal['is_lit'] = True
                    if not was_lit:
                        # Sound: crystal_activate.wav
                        self._create_spark_particles(crystal['pos'], crystal['color'], 30)
                    break
    
    def _check_termination(self):
        self.win = all(c['is_lit'] for c in self.crystals)
        if self.win:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _is_pos_vacant(self, pos):
        if pos in self.walls:
            return False
        for c in self.crystals:
            if c['pos'] == pos:
                return False
        # Also check player position
        if pos == self.player_pos:
            return False
        return True

    def _generate_level(self):
        self.walls = set()
        self.crystals = []
        self.plates = []

        # Create border walls
        for i in range(-1, self.GRID_WIDTH + 1):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_HEIGHT))
        for i in range(-1, self.GRID_HEIGHT + 1):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_WIDTH, i))

        occupied_pos = set()
        
        # Place plates and crystals
        crystal_colors = self.rng.permutation(self.CRYSTAL_COLORS).tolist()
        for i in range(self.NUM_CRYSTALS):
            # Place plate
            plate_pos = self._get_random_empty_pos(occupied_pos)
            self.plates.append({
                'pos': plate_pos,
                'crystal_id': i,
                'is_active': False,
                'color': crystal_colors[i]
            })
            occupied_pos.add(plate_pos)

            # Place corresponding crystal nearby
            crystal_pos = self._get_random_empty_pos(occupied_pos, near_pos=plate_pos, min_dist=2, max_dist=5)
            self.crystals.append({
                'pos': crystal_pos,
                'id': i,
                'is_lit': False,
                'color': crystal_colors[i]
            })
            occupied_pos.add(crystal_pos)
        
        # Place player
        self.player_pos = self._get_random_empty_pos(occupied_pos)
        occupied_pos.add(self.player_pos)
        
        # Add some random internal walls
        num_internal_walls = self.rng.integers(5, 10)
        for _ in range(num_internal_walls):
            wall_pos = self._get_random_empty_pos(occupied_pos)
            if wall_pos: # Can be None if space is tight
                self.walls.add(wall_pos)
                occupied_pos.add(wall_pos)

    def _get_random_empty_pos(self, occupied, near_pos=None, min_dist=0, max_dist=100):
        for _ in range(100): # Limit attempts to prevent infinite loops
            if near_pos:
                angle = self.rng.random() * 2 * math.pi
                dist = self.rng.uniform(min_dist, max_dist)
                x = int(near_pos[0] + math.cos(angle) * dist)
                y = int(near_pos[1] + math.sin(angle) * dist)
            else:
                x = self.rng.integers(0, self.GRID_WIDTH)
                y = self.rng.integers(0, self.GRID_HEIGHT)

            pos = (x, y)
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and pos not in occupied:
                return pos
        return None # Could not find a position

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
            "crystals_lit": sum(1 for c in self.crystals if c['is_lit']),
            "total_crystals": self.NUM_CRYSTALS,
        }

    def _world_to_screen(self, grid_pos):
        x, y = grid_pos
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)
    
    def _draw_iso_poly(self, surface, color, points):
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_block(self, surface, pos, color, height=20):
        x, y = self._world_to_screen(pos)
        points_top = [
            (x, y - height),
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF - height),
            (x, y + self.TILE_HEIGHT - height),
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF - height)
        ]
        points_left = [
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF - height),
            (x, y + self.TILE_HEIGHT - height),
            (x, y + self.TILE_HEIGHT),
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        points_right = [
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF - height),
            (x, y + self.TILE_HEIGHT - height),
            (x, y + self.TILE_HEIGHT),
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)

        self._draw_iso_poly(surface, darkest_color, points_left)
        self._draw_iso_poly(surface, darker_color, points_right)
        self._draw_iso_poly(surface, color, points_top)

    def _render_game(self):
        # --- Create a list of all drawable entities ---
        draw_list = []
        
        # Floor tiles and plates
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_plate = False
                for p in self.plates:
                    if p['pos'] == (x,y):
                        color = self.COLOR_PLATE_ACTIVE if p['is_active'] else self.COLOR_PLATE_INACTIVE
                        draw_list.append({'type': 'plate', 'pos': (x, y), 'color': color, 'active': p['is_active']})
                        is_plate = True
                        break
                if not is_plate:
                    draw_list.append({'type': 'tile', 'pos': (x, y)})
        
        # Walls, Crystals, Player
        for w_pos in self.walls:
            if 0 <= w_pos[0] < self.GRID_WIDTH and 0 <= w_pos[1] < self.GRID_HEIGHT:
                 draw_list.append({'type': 'wall', 'pos': w_pos})
        for c in self.crystals:
            # Add 'pos' to the top level for consistent sorting
            draw_list.append({'type': 'crystal', 'data': c, 'pos': c['pos']})
        draw_list.append({'type': 'player', 'pos': self.player_pos})
        
        # Sort by y-then-x for correct isometric render order
        # This key works because all items in draw_list now have a 'pos' key.
        draw_list.sort(key=lambda item: (item['pos'][0] + item['pos'][1], item['pos'][1]))

        # --- Draw everything ---
        for item in draw_list:
            item_type = item['type']
            if item_type == 'tile':
                x, y = self._world_to_screen(item['pos'])
                points = [(x, y), (x + self.TILE_W_HALF, y + self.TILE_H_HALF), (x, y + self.TILE_HEIGHT), (x - self.TILE_W_HALF, y + self.TILE_H_HALF)]
                self._draw_iso_poly(self.screen, self.COLOR_TILE, points)
            elif item_type == 'plate':
                x, y = self._world_to_screen(item['pos'])
                points = [(x, y), (x + self.TILE_W_HALF, y + self.TILE_H_HALF), (x, y + self.TILE_HEIGHT), (x - self.TILE_W_HALF, y + self.TILE_H_HALF)]
                self._draw_iso_poly(self.screen, self.COLOR_TILE, points)
                # Draw plate symbol on top
                plate_points = [
                    (x, y + self.TILE_H_HALF - 5),
                    (x + 10, y + self.TILE_HEIGHT - 10),
                    (x, y + self.TILE_HEIGHT - 5),
                    (x - 10, y + self.TILE_HEIGHT - 10)
                ]
                self._draw_iso_poly(self.screen, item['color'], plate_points)
                if item['active']:
                    pygame.gfxdraw.aacircle(self.screen, x, y + self.TILE_H_HALF, 12, item['color'])
            elif item_type == 'wall':
                self._draw_iso_block(self.screen, item['pos'], self.COLOR_WALL_TOP, height=30)
            elif item_type == 'crystal':
                c = item['data']
                color = c['color'] if c['is_lit'] else self.COLOR_CRYSTAL_INACTIVE
                self._draw_iso_block(self.screen, c['pos'], color, height=20)
                if c['is_lit']:
                    self._render_glow(c['pos'], c['color'], 20)
            elif item_type == 'player':
                self._draw_iso_block(self.screen, item['pos'], self.COLOR_PLAYER, height=15)
        
        # Draw light beams on top
        for c in self.crystals:
            if c['is_lit']:
                for p in self.plates:
                    if p['crystal_id'] == c['id'] and p['is_active']:
                        start_pos = self._world_to_screen(p['pos'])
                        end_pos = self._world_to_screen(c['pos'])
                        pygame.draw.aaline(self.screen, c['color'], 
                                           (start_pos[0], start_pos[1] + self.TILE_H_HALF), 
                                           (end_pos[0], end_pos[1]), 2)
                        break
        
        # Draw particles
        for p in self.particles:
            color_with_alpha = p['color'] + (int(p['alpha']),)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), color_with_alpha)


    def _render_glow(self, pos, color, radius):
        x, y = self._world_to_screen(pos)
        y -= 10 # Adjust for block height
        for i in range(radius // 2):
            alpha = 100 - (i * (100 // (radius // 2)))
            color_with_alpha = color + (alpha,)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius - i, color_with_alpha)
    
    def _create_spark_particles(self, pos, color, count=15):
        sx, sy = self._world_to_screen(pos)
        sy -= 10
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                'x': sx, 'y': sy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.rng.random() * 2 + 2,
                'alpha': 255,
                'color': color,
                'life': 20
            })

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['alpha'] = max(0, p['alpha'] - 12)
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _render_ui(self):
        # Lit crystals count
        lit_text = f"Lit: {sum(1 for c in self.crystals if c['is_lit'])} / {self.NUM_CRYSTALS}"
        text_surf = self.font_small.render(lit_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Steps remaining
        steps_text = f"Moves: {self.MAX_STEPS - self.steps}"
        text_surf = self.font_small.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        # Game over message
        if self.game_over:
            msg = "PUZZLE SOLVED!" if self.win else "OUT OF MOVES"
            color = self.CRYSTAL_COLORS[1] if self.win else self.CRYSTAL_COLORS[0]
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            # Draw a dark background for readability
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, bg_rect)
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup for human play
    pygame.display.init()
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cavern Crystals")
    
    # Game loop
    running = True
    while running:
        action_taken = False
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                movement = 0 # no-op
                space_pressed = 0
                
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                
                if movement or space_pressed:
                    action = [movement, space_pressed, 0] # 0 for unused shift
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    action_taken = True
                
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    action_taken = True
                if event.key == pygame.K_ESCAPE:
                    running = False

        # If no action was taken, we still need to get the latest observation for rendering
        if not action_taken:
            obs = env._get_observation()

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done and action_taken:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
        
        env.clock.tick(30) # Limit frame rate

    env.close()