
# Generated: 2025-08-27T23:39:12.785055
# Source Brief: brief_03530.md
# Brief Index: 3530

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for particles
class Particle:
    def __init__(self, x, y, color, size, lifetime, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.dx = dx
        self.dy = dy

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold space to interact with objects. Evade ghosts and find the key to unlock the exit."
    )

    game_description = (
        "Escape a procedurally generated haunted house by solving puzzles and evading ghosts within a five-minute time limit."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.FPS = 30
        
        # Colors
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_WALL = (50, 52, 70)
        self.COLOR_FLOOR = (70, 72, 90)
        self.COLOR_PLAYER = (255, 255, 100)
        self.COLOR_GHOST = (210, 70, 72)
        self.COLOR_KEY = (66, 135, 245)
        self.COLOR_DOOR_LOCKED = (180, 80, 80)
        self.COLOR_DOOR_UNLOCKED = (70, 245, 87)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (40, 42, 60, 180)

        # Game constants
        self.MAX_TIME = 300  # 5 minutes
        self.MAX_STEPS = 9000 # 300s * 30fps
        self.NUM_STAGES = 3
        self.PLAYER_MOVE_COOLDOWN_FRAMES = 4
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = 0
        self.current_stage = 1
        
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.player_has_key = False
        self.player_move_cooldown = 0
        
        self.ghosts = []
        self.key_pos = [0, 0]
        self.door_pos = [0, 0]
        self.door_locked = True
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))
        
        self.particles = []
        
        self.space_was_held = False
        self.last_player_pos = [0, 0]
        self.inactive_steps = 0

        self.reward_buffer = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_TIME
        self.current_stage = 1
        self.reward_buffer = 0.0
        
        self._generate_stage()
        
        self.space_was_held = False
        self.last_player_pos = list(self.player_pos)
        self.inactive_steps = 0
        
        return self._get_observation(), self._get_info()

    def _generate_stage(self):
        # 1. Create grid with walls
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # 2. Carve a main room
        self.grid[1:self.GRID_WIDTH-1, 1:self.GRID_HEIGHT-1] = 0

        # 3. Add random internal walls, ensuring connectivity
        for _ in range(self.np_random.integers(5, 15)):
            wall_x = self.np_random.integers(2, self.GRID_WIDTH - 2)
            wall_y = self.np_random.integers(2, self.GRID_HEIGHT - 2)
            length = self.np_random.integers(3, 8)
            if self.np_random.random() > 0.5: # Horizontal wall
                self.grid[wall_x:min(self.GRID_WIDTH-2, wall_x+length), wall_y] = 1
            else: # Vertical wall
                self.grid[wall_x, wall_y:min(self.GRID_HEIGHT-2, wall_y+length)] = 1
        
        # Ensure connectivity using flood fill
        q = [(2, 2)]
        reachable = set(q)
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0 and (nx, ny) not in reachable:
                    reachable.add((nx,ny))
                    q.append((nx,ny))
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0 and (x, y) not in reachable:
                    self.grid[x, y] = 1
        
        # 4. Find valid spawn points
        valid_spawns = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if self.grid[x, y] == 0]
        
        # 5. Place entities
        random.shuffle(valid_spawns)
        
        self.player_pos = list(valid_spawns.pop())
        self.player_visual_pos = [self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE]
        self.last_player_pos = list(self.player_pos)
        self.player_has_key = False
        
        self.key_pos = list(valid_spawns.pop())
        
        # Place door on a wall adjacent to an empty space
        door_candidates = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                if self.grid[x,y] == 1:
                    # Check if it has empty neighbors
                    if (self.grid[x-1,y] == 0 or self.grid[x+1,y] == 0 or self.grid[x,y-1] == 0 or self.grid[x,y+1] == 0):
                        door_candidates.append((x,y))
        self.door_pos = list(random.choice(door_candidates)) if door_candidates else [self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2]

        self.door_locked = True
        
        # 6. Place ghosts
        self.ghosts = []
        base_speed = 0.5 + (self.current_stage - 1) * 0.05  # tiles per second
        for _ in range(3):
            path_len = self.np_random.integers(4, 10)
            start_pos = list(valid_spawns.pop())
            
            # Find a valid patrol path
            end_pos = start_pos
            if self.np_random.random() > 0.5: # Horizontal
                for i in range(path_len, 0, -1):
                    if start_pos[0] + i < self.GRID_WIDTH and self.grid[start_pos[0]+i, start_pos[1]] == 0:
                        end_pos = [start_pos[0]+i, start_pos[1]]
                        break
            else: # Vertical
                for i in range(path_len, 0, -1):
                    if start_pos[1] + i < self.GRID_HEIGHT and self.grid[start_pos[0], start_pos[1]+i] == 0:
                        end_pos = [start_pos[0], start_pos[1]+i]
                        break
            
            self.ghosts.append({
                "pos": [float(start_pos[0]), float(start_pos[1])],
                "visual_pos": [start_pos[0] * self.TILE_SIZE, start_pos[1] * self.TILE_SIZE],
                "path": [start_pos, end_pos],
                "target_idx": 1,
                "speed": base_speed / self.FPS # tiles per frame
            })

        self.particles = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_buffer = 0.0

        # --- Update state ---
        self.timer = max(0, self.timer - 1 / self.FPS)
        self.reward_buffer += 0.1 / self.FPS  # Survival reward

        # Player movement
        movement = action[0]
        self.player_move_cooldown = max(0, self.player_move_cooldown - 1)
        if movement != 0 and self.player_move_cooldown == 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT and self.grid[new_x, new_y] == 0:
                self.player_pos = [new_x, new_y]
                self.player_move_cooldown = self.PLAYER_MOVE_COOLDOWN_FRAMES
                # sfx: player_step.wav

        # Player interaction
        space_held = action[1] == 1
        interact_triggered = space_held and not self.space_was_held
        if interact_triggered:
            # Check for key pickup
            if self.player_pos == self.key_pos and not self.player_has_key:
                self.player_has_key = True
                self.key_pos = [-1, -1] # Move key off-screen
                self.reward_buffer += 5.0
                self.score += 50
                # sfx: key_pickup.wav
                for _ in range(30):
                    self.particles.append(Particle(self.player_visual_pos[0] + self.TILE_SIZE/2, self.player_visual_pos[1] + self.TILE_SIZE/2, self.COLOR_KEY, 5, 20, (self.np_random.random()-0.5)*4, (self.np_random.random()-0.5)*4))

            # Check for door unlock
            dist_to_door = math.hypot(self.player_pos[0] - self.door_pos[0], self.player_pos[1] - self.door_pos[1])
            if dist_to_door < 1.5 and self.player_has_key and self.door_locked:
                self.door_locked = False
                self.reward_buffer += 5.0 # Puzzle solved reward
                self.score += 50
                # sfx: door_unlock.wav
                for _ in range(30):
                    self.particles.append(Particle(self.door_pos[0]*self.TILE_SIZE + self.TILE_SIZE/2, self.door_pos[1]*self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_DOOR_UNLOCKED, 5, 20, (self.np_random.random()-0.5)*4, (self.np_random.random()-0.5)*4))

        self.space_was_held = space_held

        # Ghost movement and collision
        for ghost in self.ghosts:
            target = ghost["path"][ghost["target_idx"]]
            direction = [target[0] - ghost["pos"][0], target[1] - ghost["pos"][1]]
            dist = math.hypot(*direction)
            
            if dist < ghost["speed"]:
                ghost["pos"] = [float(target[0]), float(target[1])]
                ghost["target_idx"] = 1 - ghost["target_idx"] # Flip target
            else:
                norm_dir = [d / dist for d in direction]
                ghost["pos"][0] += norm_dir[0] * ghost["speed"]
                ghost["pos"][1] += norm_dir[1] * ghost["speed"]
            
            if self.np_random.random() < 0.2: # Ghost particle trail
                self.particles.append(Particle(ghost["pos"][0]*self.TILE_SIZE + self.TILE_SIZE/2, ghost["pos"][1]*self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_GHOST, 3, 15, (self.np_random.random()-0.5)*0.5, (self.np_random.random()-0.5)*0.5))

            if int(ghost["pos"][0]) == self.player_pos[0] and int(ghost["pos"][1]) == self.player_pos[1]:
                self.game_over = True
                self.reward_buffer = -100.0
                self.score -= 100
                # sfx: player_caught.wav

        # Update visual positions for smooth interpolation
        lerp_rate = 0.25
        self.player_visual_pos[0] += (self.player_pos[0] * self.TILE_SIZE - self.player_visual_pos[0]) * lerp_rate
        self.player_visual_pos[1] += (self.player_pos[1] * self.TILE_SIZE - self.player_visual_pos[1]) * lerp_rate
        for ghost in self.ghosts:
            ghost["visual_pos"][0] += (ghost["pos"][0] * self.TILE_SIZE - ghost["visual_pos"][0]) * lerp_rate
            ghost["visual_pos"][1] += (ghost["pos"][1] * self.TILE_SIZE - ghost["visual_pos"][1]) * lerp_rate

        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        # Inactivity penalty
        if self.player_pos == self.last_player_pos:
            self.inactive_steps += 1
            if self.inactive_steps == 6 * 30: # 6 seconds, to be less punishing
                self.reward_buffer -= 0.2
                self.inactive_steps = 0 # Reset after penalty
        else:
            self.inactive_steps = 0
        self.last_player_pos = list(self.player_pos)

        # --- Check for termination / progression ---
        terminated = False
        if self.game_over:
            terminated = True
        elif self.timer <= 0:
            terminated = True
            self.game_over = True
            self.reward_buffer = -50.0
            self.score -= 50
            # sfx: timeout.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        elif self.player_pos == self.door_pos and not self.door_locked:
            if self.current_stage < self.NUM_STAGES:
                self.current_stage += 1
                self.reward_buffer += 10.0
                self.score += 100
                self._generate_stage()
                # sfx: stage_complete.wav
            else:
                self.win = True
                self.game_over = True
                terminated = True
                self.reward_buffer = 100.0
                self.score += 200
                # sfx: game_win.wav

        self.score += self.reward_buffer # Add per-step rewards to total score
        
        return (
            self._get_observation(),
            self.reward_buffer,
            terminated,
            False,
            self._get_info()
        )

    def _draw_glow(self, surface, pos, color, radius, intensity=0.1):
        for i in range(radius, 0, -2):
            alpha = int(255 * (1 - i / radius) * intensity)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], i, color + (alpha,))

    def _get_observation(self):
        # Clear screen with a flickering effect
        flicker = self.np_random.integers(0, 5)
        bg_color = (max(0, self.COLOR_BG[0] - flicker), max(0, self.COLOR_BG[1] - flicker), max(0, self.COLOR_BG[2] - flicker))
        self.screen.fill(bg_color)
        
        # Render game elements
        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.grid[x, y] == 1:
                    color = self.COLOR_WALL
                else:
                    color = self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw door
        door_color = self.COLOR_DOOR_UNLOCKED if not self.door_locked else self.COLOR_DOOR_LOCKED
        door_rect = (self.door_pos[0] * self.TILE_SIZE, self.door_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, door_color, door_rect)
        if not self.door_locked:
            self._draw_glow(self.screen, (door_rect[0] + self.TILE_SIZE//2, door_rect[1] + self.TILE_SIZE//2), (0, 255, 0), 20)

        # Draw key
        if not self.player_has_key and self.key_pos[0] != -1:
            key_center_x = self.key_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
            key_center_y = self.key_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
            pygame.draw.circle(self.screen, self.COLOR_KEY, (key_center_x, key_center_y), self.TILE_SIZE // 3)
            self._draw_glow(self.screen, (key_center_x, key_center_y), self.COLOR_KEY, 15)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw ghosts
        for ghost in self.ghosts:
            pos_x = int(ghost["visual_pos"][0] + self.TILE_SIZE // 2)
            pos_y = int(ghost["visual_pos"][1] + self.TILE_SIZE // 2)
            pygame.draw.circle(self.screen, self.COLOR_GHOST, (pos_x, pos_y), self.TILE_SIZE // 2)
            self._draw_glow(self.screen, (pos_x, pos_y), self.COLOR_GHOST, 25)

        # Draw player
        player_center_x = int(self.player_visual_pos[0] + self.TILE_SIZE // 2)
        player_center_y = int(self.player_visual_pos[1] + self.TILE_SIZE // 2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]), self.TILE_SIZE, self.TILE_SIZE), border_radius=3)
        self._draw_glow(self.screen, (player_center_x, player_center_y), self.COLOR_PLAYER, 30, intensity=0.2)
        if self.player_has_key:
             pygame.draw.circle(self.screen, self.COLOR_KEY, (player_center_x, player_center_y), 3)


        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Stage Text
        stage_text = self.font_small.render(f"Stage: {self.current_stage} / {self.NUM_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        # Timer Text
        mins, secs = divmod(int(self.timer), 60)
        timer_text = self.font_small.render(f"Time: {mins:02d}:{secs:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Score Text
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.win:
                msg = "YOU ESCAPED!"
                color = self.COLOR_DOOR_UNLOCKED
            else:
                msg = "YOU WERE CAUGHT"
                color = self.COLOR_GHOST
                
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "timer": self.timer,
            "player_has_key": self.player_has_key,
        }

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
        
        # Test assertions from brief
        assert 0 <= self.player_pos[0] < self.GRID_WIDTH and 0 <= self.player_pos[1] < self.GRID_HEIGHT
        assert 0 <= self.timer <= self.MAX_TIME
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # To render the game window
    pygame.display.set_caption("Haunted House Escape")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Manual control mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        # --- Human Controls ---
        movement_action = 0
        space_action = 0
        shift_action = 0 # Unused in this game

        keys = pygame.key.get_pressed()
        for key, move in key_map.items():
            if keys[key]:
                movement_action = move
                break
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, shift_action]
        # --- End Human Controls ---

        # action = env.action_space.sample() # For random agent

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()