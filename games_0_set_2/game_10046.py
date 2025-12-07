import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:49:26.917141
# Source Brief: brief_00046.md
# Brief Index: 46
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a surreal dream-maze, collecting fragments to craft tools and unlock the path to the exit. "
        "Evade the ever-watchful guards that patrol the corridors."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Press space to collect fragments or use tools. "
        "Hold shift and press space to craft new tools from collected fragments."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAZE_COLS = 20
        self.MAZE_ROWS = 12
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.MAZE_COLS
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.MAZE_ROWS

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Colors (Surreal, dreamlike palette)
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_WALLS = (80, 60, 120)
        self.COLOR_PLAYER = (255, 255, 100)
        self.COLOR_FRAGMENT = (0, 180, 255)
        self.COLOR_GUARD = (255, 50, 100)
        self.COLOR_EXIT = (180, 0, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_KEY = (100, 255, 100)
        self.COLOR_LOCKPICK = (200, 150, 255)
        self.COLOR_TDD = (255, 150, 50)
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_health = None
        self.inventory = None
        self.maze = None
        self.guards = None
        self.fragments = None
        self.door_pos = None
        self.door_locked = None
        self.exit_pos = None
        self.particles = None
        self.interaction_cooldown = None
        self.guard_effects = None
        
        # Game parameters
        self.MAX_STEPS = 2000
        self.PLAYER_SPEED = 4
        self.INITIAL_GUARD_SPEED = 0.5
        self.MAX_HEALTH = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.MAX_HEALTH
        self.inventory = {"fragments": 0, "key": 0, "lockpick": 0, "tdd": 0}
        self.interaction_cooldown = 0
        self.guard_effects = {}

        self._generate_maze_and_place_objects()
        
        self.particles = deque(maxlen=200)

        return self._get_observation(), self._get_info()

    def _generate_maze_and_place_objects(self):
        # 1. Initialize grid
        self.maze = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(self.MAZE_COLS)] for _ in range(self.MAZE_ROWS)]

        # 2. Recursive backtracker algorithm
        def carve_passages(cx, cy):
            self.maze[cy][cx]['visited'] = True
            directions = ['N', 'S', 'E', 'W']
            self.np_random.shuffle(directions)
            for direction in directions:
                nx, ny = cx, cy
                if direction == 'N': ny -= 1
                if direction == 'S': ny += 1
                if direction == 'E': nx += 1
                if direction == 'W': nx -= 1

                if 0 <= nx < self.MAZE_COLS and 0 <= ny < self.MAZE_ROWS and not self.maze[ny][nx]['visited']:
                    if direction == 'N': self.maze[cy][cx]['N'], self.maze[ny][nx]['S'] = False, False
                    if direction == 'S': self.maze[cy][cx]['S'], self.maze[ny][nx]['N'] = False, False
                    if direction == 'E': self.maze[cy][cx]['E'], self.maze[ny][nx]['W'] = False, False
                    if direction == 'W': self.maze[cy][cx]['W'], self.maze[ny][nx]['E'] = False, False
                    carve_passages(nx, ny)

        carve_passages(0, 0)

        # 3. Place objects
        self.player_pos = pygame.Vector2(self.CELL_WIDTH * 0.5, self.CELL_HEIGHT * 0.5)
        self.exit_pos = (self.MAZE_COLS - 1, self.MAZE_ROWS - 1)
        
        # Place door on the solution path
        path = self._solve_maze((0,0), self.exit_pos)
        if path and len(path) > 4:
            self.door_pos = path[len(path) // 2]
            self.door_locked = True
        else: # Failsafe
            self.door_pos = (self.MAZE_COLS // 2, self.MAZE_ROWS // 2)
            self.door_locked = True

        # Place fragments and guards in dead ends
        dead_ends = []
        for r in range(self.MAZE_ROWS):
            for c in range(self.MAZE_COLS):
                if sum(self.maze[r][c][d] for d in 'NSEW') == 3 and (c,r) != (0,0):
                    dead_ends.append((c,r))
        self.np_random.shuffle(dead_ends)
        
        num_fragments_to_place = 10
        self.fragments = [pygame.Vector2(c * self.CELL_WIDTH + self.CELL_WIDTH/2, r * self.CELL_HEIGHT + self.CELL_HEIGHT/2) for c,r in dead_ends[:num_fragments_to_place]]
        
        num_guards = 3
        guard_placements = dead_ends[num_fragments_to_place:num_fragments_to_place+num_guards]
        self.guards = []
        for c, r in guard_placements:
            patrol_path = self._find_patrol_path((c, r))
            if patrol_path:
                self.guards.append({
                    "pos": pygame.Vector2(c * self.CELL_WIDTH + self.CELL_WIDTH/2, r * self.CELL_HEIGHT + self.CELL_HEIGHT/2),
                    "path": [pygame.Vector2(px * self.CELL_WIDTH + self.CELL_WIDTH/2, py * self.CELL_HEIGHT + self.CELL_HEIGHT/2) for px,py in patrol_path],
                    "target_idx": 1,
                    "id": len(self.guards)
                })

    def _solve_maze(self, start_cell, end_cell):
        q = deque([(start_cell, [start_cell])])
        visited = {start_cell}
        while q:
            (c, r), path = q.popleft()
            if (c, r) == end_cell:
                return path
            
            # North
            if not self.maze[r][c]['N'] and (c, r-1) not in visited:
                visited.add((c, r-1))
                q.append(((c, r-1), path + [(c, r-1)]))
            # South
            if not self.maze[r][c]['S'] and (c, r+1) not in visited:
                visited.add((c, r+1))
                q.append(((c, r+1), path + [(c, r+1)]))
            # East
            if not self.maze[r][c]['E'] and (c+1, r) not in visited:
                visited.add((c+1, r))
                q.append(((c+1, r), path + [(c+1, r)]))
            # West
            if not self.maze[r][c]['W'] and (c-1, r) not in visited:
                visited.add((c-1, r))
                q.append(((c-1, r), path + [(c-1, r)]))
        return None

    def _find_patrol_path(self, start_cell):
        # Find a path from a dead end out to a junction
        path = [start_cell]
        c, r = start_cell
        while sum(self.maze[r][c][d] for d in 'NSEW') >= 2: # while not a junction or start
            possible_moves = []
            prev_cell = path[-2] if len(path) > 1 else None
            if not self.maze[r][c]['N'] and (c, r-1) != prev_cell: possible_moves.append((c, r - 1))
            if not self.maze[r][c]['S'] and (c, r+1) != prev_cell: possible_moves.append((c, r + 1))
            if not self.maze[r][c]['E'] and (c+1, r) != prev_cell: possible_moves.append((c + 1, r))
            if not self.maze[r][c]['W'] and (c-1, r) != prev_cell: possible_moves.append((c - 1, r))

            if not possible_moves: break
            c, r = possible_moves[0]
            path.append((c,r))
            if len(path) > 10: break # cap path length
        return path

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        events = []

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_held = action[2] == 1
        
        # Cooldowns
        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
        
        # Update game logic
        self._handle_player_movement(movement)
        
        if space_pressed and self.interaction_cooldown == 0:
            self._handle_interaction(shift_held, events)

        self._update_guards(events)
        self._update_particles()
        
        # Calculate rewards from events
        reward += self._calculate_reward(events)
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            if self.player_health <= 0:
                reward += -50.0
                self.score -= 50.0
            elif truncated and not terminated:
                reward += -25.0
                self.score -= 25.0
            elif terminated: # Reached exit
                reward += 100.0
                self.score += 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_player_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            move_vec.scale_to_length(self.PLAYER_SPEED)
            new_pos = self.player_pos + move_vec

            # Wall collision
            player_radius = 8
            current_cell_x = int(self.player_pos.x / self.CELL_WIDTH)
            current_cell_y = int(self.player_pos.y / self.CELL_HEIGHT)
            
            current_cell_x = max(0, min(self.MAZE_COLS - 1, current_cell_x))
            current_cell_y = max(0, min(self.MAZE_ROWS - 1, current_cell_y))

            # X-axis collision
            if new_pos.x < self.player_pos.x: # Moving left
                if self.maze[current_cell_y][current_cell_x]['W'] and new_pos.x - player_radius < current_cell_x * self.CELL_WIDTH:
                    new_pos.x = self.player_pos.x
            elif new_pos.x > self.player_pos.x: # Moving right
                if self.maze[current_cell_y][current_cell_x]['E'] and new_pos.x + player_radius > (current_cell_x + 1) * self.CELL_WIDTH:
                    new_pos.x = self.player_pos.x

            # Y-axis collision
            if new_pos.y < self.player_pos.y: # Moving up
                if self.maze[current_cell_y][current_cell_x]['N'] and new_pos.y - player_radius < current_cell_y * self.CELL_HEIGHT:
                    new_pos.y = self.player_pos.y
            elif new_pos.y > self.player_pos.y: # Moving down
                if self.maze[current_cell_y][current_cell_x]['S'] and new_pos.y + player_radius > (current_cell_y + 1) * self.CELL_HEIGHT:
                    new_pos.y = self.player_pos.y
            
            self.player_pos = new_pos
            
            # Add trail particles
            self.particles.append({
                "pos": self.player_pos.copy(),
                "vel": move_vec * -0.1,
                "radius": self.np_random.uniform(2, 4),
                "lifetime": 20,
                "color": self.COLOR_PLAYER
            })

    def _handle_interaction(self, shift_held, events):
        self.interaction_cooldown = 10 # 10 frames cooldown

        if shift_held: # Crafting
            if self.inventory["fragments"] >= 3 and self.inventory["key"] == 0:
                self.inventory["fragments"] -= 3
                self.inventory["key"] += 1
                events.append("crafted_tool")
            elif self.inventory["fragments"] >= 5 and self.inventory["lockpick"] < 2:
                self.inventory["fragments"] -= 5
                self.inventory["lockpick"] += 1
                events.append("crafted_tool")
            elif self.inventory["fragments"] >= 5 and self.inventory["tdd"] < 2:
                self.inventory["fragments"] -= 5
                self.inventory["tdd"] += 1
                events.append("crafted_tool")
        else: # Interaction
            for frag in self.fragments[:]:
                if self.player_pos.distance_to(frag) < 20:
                    self.fragments.remove(frag)
                    self.inventory["fragments"] += 1
                    events.append("fragment_collected")
                    break
            
            door_center = pygame.Vector2(self.door_pos[0]*self.CELL_WIDTH+self.CELL_WIDTH/2, self.door_pos[1]*self.CELL_HEIGHT+self.CELL_HEIGHT/2)
            if self.door_locked and self.inventory["key"] > 0 and self.player_pos.distance_to(door_center) < 30:
                self.door_locked = False
                self.inventory["key"] -= 1
                events.append("door_unlocked")
            
            for guard in self.guards:
                if self.player_pos.distance_to(guard["pos"]) < 40:
                    if self.inventory["lockpick"] > 0:
                        self.inventory["lockpick"] -= 1
                        self.guard_effects[guard["id"]] = {"type": "disabled", "duration": 150} # 5 seconds
                        events.append("used_tool")
                        break
                    elif self.inventory["tdd"] > 0:
                        self.inventory["tdd"] -= 1
                        self.guard_effects[guard["id"]] = {"type": "slowed", "duration": 300} # 10 seconds
                        events.append("used_tool")
                        break

    def _update_guards(self, events):
        guard_speed = self.INITIAL_GUARD_SPEED + (self.steps // 500) * 0.1
        min_dist_to_guard = float('inf')

        for guard in self.guards:
            dist = self.player_pos.distance_to(guard["pos"])
            min_dist_to_guard = min(min_dist_to_guard, dist)

            if guard["id"] in self.guard_effects:
                effect = self.guard_effects[guard["id"]]
                effect["duration"] -= 1
                if effect["duration"] <= 0:
                    del self.guard_effects[guard["id"]]
            
            effect_type = self.guard_effects.get(guard["id"], {}).get("type")
            
            current_speed = guard_speed
            if effect_type == "disabled": current_speed = 0
            elif effect_type == "slowed": current_speed *= 0.3

            if current_speed > 0 and guard["path"]:
                target_pos = guard["path"][guard["target_idx"]]
                direction = (target_pos - guard["pos"])
                
                if direction.length() < current_speed * 2:
                    guard["pos"] = target_pos
                    guard["target_idx"] = (guard["target_idx"] + 1) % len(guard["path"])
                else:
                    direction.scale_to_length(current_speed)
                    guard["pos"] += direction
            
            if effect_type != "disabled" and self.player_pos.distance_to(guard["pos"]) < 15:
                self.player_health = max(0, self.player_health - 20)
                events.append("caught_by_guard")
                push_vec = self.player_pos - guard["pos"]
                if push_vec.length() > 0:
                    push_vec.scale_to_length(15)
                    self.player_pos += push_vec

        if min_dist_to_guard < 150:
            events.append(("near_guard", min_dist_to_guard))

    def _update_particles(self):
        for p in list(self.particles):
            p["pos"] += p["vel"]
            p["radius"] -= 0.05
            p["lifetime"] -= 1
            if p["lifetime"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self, events):
        reward = -0.001
        for event in events:
            if isinstance(event, tuple):
                event_type, value = event
                if event_type == "near_guard":
                    reward -= 0.1 / (1 + value / 20.0)
            else:
                if event == "fragment_collected": reward += 0.5
                if event == "crafted_tool": reward += 5.0
                if event == "door_unlocked": reward += 10.0
                if event == "used_tool": reward += 2.0
                if event == "caught_by_guard": reward -= 1.0
        return reward

    def _check_termination(self):
        exit_cell_x, exit_cell_y = self.exit_pos
        exit_center = pygame.Vector2(exit_cell_x * self.CELL_WIDTH + self.CELL_WIDTH/2, exit_cell_y * self.CELL_HEIGHT + self.CELL_HEIGHT/2)
        
        if self.player_pos.distance_to(exit_center) < self.CELL_WIDTH / 2:
            return True
        if self.player_health <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.MAZE_ROWS):
            for c in range(self.MAZE_COLS):
                x, y = c * self.CELL_WIDTH, r * self.CELL_HEIGHT
                if self.maze[r][c]['N']: pygame.draw.line(self.screen, self.COLOR_WALLS, (x, y), (x + self.CELL_WIDTH, y), 2)
                if self.maze[r][c]['S']: pygame.draw.line(self.screen, self.COLOR_WALLS, (x, y + self.CELL_HEIGHT), (x + self.CELL_WIDTH, y + self.CELL_HEIGHT), 2)
                if self.maze[r][c]['E']: pygame.draw.line(self.screen, self.COLOR_WALLS, (x + self.CELL_WIDTH, y), (x + self.CELL_WIDTH, y + self.CELL_HEIGHT), 2)
                if self.maze[r][c]['W']: pygame.draw.line(self.screen, self.COLOR_WALLS, (x, y), (x, y + self.CELL_HEIGHT), 2)
        
        for p in self.particles:
            if p["radius"] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*p["color"], max(0, min(255, p["lifetime"]*10))))

        exit_x = self.exit_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        exit_y = self.exit_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        self._draw_glowing_circle(self.screen, (exit_x, exit_y), self.COLOR_EXIT, 15, 5)
        
        for frag_pos in self.fragments:
            self._draw_glowing_circle(self.screen, frag_pos, self.COLOR_FRAGMENT, 8, 4)
            
        if self.door_locked:
            dx, dy = self.door_pos
            door_rect = pygame.Rect(dx*self.CELL_WIDTH+5, dy*self.CELL_HEIGHT+5, self.CELL_WIDTH-10, self.CELL_HEIGHT-10)
            self._draw_glowing_rect(self.screen, door_rect, self.COLOR_KEY, 3)

        for guard in self.guards:
            effect = self.guard_effects.get(guard["id"])
            if effect and effect["type"] == "disabled":
                self._draw_glowing_circle(self.screen, guard["pos"], (100, 100, 100), 10, 3)
            else:
                self._draw_glowing_circle(self.screen, guard["pos"], self.COLOR_GUARD, 10, 5 + int(math.sin(self.steps * 0.2) * 2))
                if effect and effect["type"] == "slowed":
                    pygame.gfxdraw.aacircle(self.screen, int(guard["pos"].x), int(guard["pos"].y), 15, self.COLOR_TDD)

        aura_size = int(12 * (self.player_health / self.MAX_HEALTH))
        self._draw_glowing_circle(self.screen, self.player_pos, self.COLOR_PLAYER, aura_size, 8)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        health_text = self.font_small.render(f"AURA: {self.player_health}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))

        base_x = 10
        base_y = self.SCREEN_HEIGHT - 30
        
        frag_text = self.font_small.render(f"{self.inventory['fragments']}", True, self.COLOR_UI_TEXT)
        self._draw_glowing_circle(self.screen, (base_x + 10, base_y + 10), self.COLOR_FRAGMENT, 8, 2)
        self.screen.blit(frag_text, (base_x + 25, base_y))

        tool_x = base_x + 70
        if self.inventory['key'] > 0:
            self._draw_glowing_rect(self.screen, pygame.Rect(tool_x, base_y, 20, 20), self.COLOR_KEY, 2)
            tool_x += 30
        if self.inventory['lockpick'] > 0:
            self._draw_glowing_rect(self.screen, pygame.Rect(tool_x, base_y, 20, 20), self.COLOR_LOCKPICK, 2)
            tool_x += 30
        if self.inventory['tdd'] > 0:
            self._draw_glowing_rect(self.screen, pygame.Rect(tool_x, base_y, 20, 20), self.COLOR_TDD, 2)

    def _draw_glowing_circle(self, surface, pos, color, radius, glow_size):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(glow_size, 0, -1):
            alpha = 150 * (1 - (i / glow_size))**2
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius + i), (*color, int(alpha)))
        if radius > 0:
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)

    def _draw_glowing_rect(self, surface, rect, color, glow_size):
        for i in range(glow_size, 0, -1):
            alpha = 150 * (1 - (i / glow_size))**2
            glow_rect = rect.inflate(i*2, i*2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*color, int(alpha)), shape_surf.get_rect(), border_radius=3)
            surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "inventory": self.inventory,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Dream Thief")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                 space_action = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            reset_choice = input("Play again? (y/n): ")
            if reset_choice.lower() == 'y':
                obs, info = env.reset()
                terminated = False
                truncated = False
            else:
                break

    env.close()