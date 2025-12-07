
# Generated: 2025-08-27T23:07:28.177133
# Source Brief: brief_03354.md
# Brief Index: 3354

        
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

    user_guide = (
        "Controls: Use arrow keys to 'push' blocks from the center of the grid. "
        "A push travels outwards and moves the first block it hits."
    )

    game_description = (
        "A fast-paced puzzle game. Push colored blocks onto their matching goals before time runs out. "
        "Complete three increasingly difficult stages to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.CENTER_X, self.CENTER_Y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        
        self.STAGE_CONFIG = {
            1: {"blocks": 3, "time": 60},
            2: {"blocks": 6, "time": 60},
            3: {"blocks": 10, "time": 60}
        }
        self.MAX_STAGES = 3
        self.MAX_STEPS = sum(v['time'] for v in self.STAGE_CONFIG.values()) * self.FPS

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 48, 56)
        self.COLOR_WALL = (80, 85, 97)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.BLOCK_COLORS = [
            (255, 95, 95), (95, 255, 95), (95, 175, 255), (255, 255, 95),
            (255, 95, 255), (95, 255, 255), (255, 165, 0), (128, 0, 128),
            (0, 128, 128), (128, 128, 0)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.stage_timer = 0
        self.blocks = []
        self.goals = []
        self.walls = []
        self.particles = []
        self.push_effect = None
        self.last_action_time = 0
        self.action_cooldown = int(self.FPS * 0.3) # Cooldown for push action

        # Initialize and validate
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.particles = []
        self.push_effect = None
        self.last_action_time = 0
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Generates a new puzzle layout for the current stage."""
        self.blocks = []
        self.goals = []
        self.walls = set()
        
        # Reset stage timer
        self.stage_timer = self.STAGE_CONFIG[self.current_stage]["time"] * self.FPS

        # Create walls around the border
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        num_blocks = self.STAGE_CONFIG[self.current_stage]["blocks"]
        
        # Generate solvable puzzle by reverse moves
        occupied_coords = set(self.walls)
        
        # Place goals
        goal_candidates = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                goal_candidates.append((x, y))
        
        self.np_random.shuffle(goal_candidates)
        
        for i in range(num_blocks):
            pos = goal_candidates.pop(0)
            color = self.BLOCK_COLORS[i]
            self.goals.append({"pos": pos, "color": color})
            occupied_coords.add(pos)
            
        # Place blocks on goals initially
        for i, goal in enumerate(self.goals):
            pos = goal["pos"]
            self.blocks.append({
                "id": i,
                "grid_pos": pos,
                "pixel_pos": pygame.Vector2(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE),
                "anim_progress": 1.0,
                "color": goal["color"]
            })

        # Scramble blocks with reverse moves
        scramble_steps = num_blocks * 5
        for _ in range(scramble_steps):
            block = self.np_random.choice(self.blocks)
            
            # Try a "pull" move (inverse of push)
            dx, dy = self.np_random.choice([ (0, 1), (0, -1), (1, 0), (-1, 0) ])
            
            current_pos = block["grid_pos"]
            new_pos = (current_pos[0] - dx, current_pos[1] - dy)
            
            if new_pos not in occupied_coords:
                occupied_coords.remove(current_pos)
                block["grid_pos"] = new_pos
                occupied_coords.add(new_pos)

        # Finalize block pixel positions after scrambling
        for block in self.blocks:
            pos = block["grid_pos"]
            block["pixel_pos"] = pygame.Vector2(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE)


    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = self.game_over

        if terminated:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.stage_timer -= 1

        # --- Handle Push Action ---
        is_animating = any(b["anim_progress"] < 1.0 for b in self.blocks)
        can_act = self.steps > self.last_action_time + self.action_cooldown

        if movement != 0 and not is_animating and can_act:
            self.last_action_time = self.steps
            reward += self._handle_push(movement)
        
        # --- Update Game Logic ---
        self._update_animations()

        # --- Check for Stage Completion ---
        if self._are_all_blocks_on_goals():
            self.score += 50 # Stage clear bonus
            reward += 50
            # SFX: Stage Clear
            
            for block in self.blocks:
                self._create_particles(block["pixel_pos"] + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2), block["color"], 20)

            self.current_stage += 1
            if self.current_stage > self.MAX_STAGES:
                self.game_over = True
                terminated = True
                self.score += 100 # Game win bonus
                reward += 100
            else:
                self._setup_stage()
        
        # --- Check for Timeout ---
        if self.stage_timer <= 0:
            self.game_over = True
            terminated = True
            self.score -= 100 # Timeout penalty
            reward -= 100
            # SFX: Game Over
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            if not self._are_all_blocks_on_goals():
                 self.score -= 100
                 reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement_action):
        """Implements the 'push from center' mechanic."""
        reward = -0.01 # Small cost for taking an action
        
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        dx, dy = direction_map[movement_action]

        ray_coords = []
        if dx == 0: # Vertical push
            for y in range(self.CENTER_Y + dy, self.GRID_HEIGHT if dy > 0 else -1, dy):
                ray_coords.append((self.CENTER_X, y))
        else: # Horizontal push
            for x in range(self.CENTER_X + dx, self.GRID_WIDTH if dx > 0 else -1, dx):
                ray_coords.append((x, self.CENTER_Y))
        
        self.push_effect = {"dir": (dx, dy), "life": int(self.FPS * 0.2)}
        # SFX: Push attempt

        block_positions = {b["grid_pos"]: b for b in self.blocks}
        
        for pos in ray_coords:
            if pos in block_positions:
                block_to_push = block_positions[pos]
                
                # Check destination
                old_pos = block_to_push["grid_pos"]
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                
                occupied_tiles = self.walls.union(set(b["grid_pos"] for b in self.blocks))
                
                if new_pos not in occupied_tiles:
                    # Valid push
                    # SFX: Block Slide
                    goal = self.goals[block_to_push["id"]]
                    
                    dist_before = math.hypot(old_pos[0] - goal["pos"][0], old_pos[1] - goal["pos"][1])
                    dist_after = math.hypot(new_pos[0] - goal["pos"][0], new_pos[1] - goal["pos"][1])

                    if dist_after < dist_before:
                        reward += 0.5 # Moved closer to goal
                    else:
                        reward -= 0.5 # Moved away from goal

                    block_to_push["grid_pos"] = new_pos
                    block_to_push["anim_progress"] = 0.0

                    if new_pos == goal["pos"]:
                        reward += 5 # Placed on goal!
                        self.score += 5
                        # SFX: Goal achieved
                        self._create_particles(pygame.Vector2(new_pos) * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2), block_to_push["color"], 10)

                return reward # Stop after interacting with the first block
        
        return reward

    def _are_all_blocks_on_goals(self):
        for block in self.blocks:
            if block["grid_pos"] != self.goals[block["id"]]["pos"]:
                return False
        return True

    def _update_animations(self):
        # Update blocks
        for block in self.blocks:
            if block["anim_progress"] < 1.0:
                block["anim_progress"] = min(1.0, block["anim_progress"] + 0.15) # Animation speed
                
                start_pos = (pygame.Vector2(block["grid_pos"]) - pygame.Vector2(block["grid_pos"]) / pygame.Vector2(block["grid_pos"]).length() if block["grid_pos"]!=(0,0) and block["grid_pos"][0] != block["grid_pos"][1] else pygame.Vector2(block["grid_pos"][0]-1, block["grid_pos"][1]-1)) * self.TILE_SIZE if "start_anim_pos" not in block else block["start_anim_pos"]
                
                target_pos = pygame.Vector2(block["grid_pos"]) * self.TILE_SIZE
                
                # Ease-out interpolation
                t = 1 - (1 - block["anim_progress"])**3
                block["pixel_pos"] = start_pos.lerp(target_pos, t)
            else:
                 start_pos = pygame.Vector2(block["grid_pos"]) * self.TILE_SIZE
                 block["pixel_pos"] = start_pos
                 block["start_anim_pos"] = start_pos


        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)

        # Update push effect
        if self.push_effect:
            self.push_effect["life"] -= 1
            if self.push_effect["life"] <= 0:
                self.push_effect = None

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "radius": self.np_random.uniform(3, 7),
                "color": color,
                "life": self.np_random.integers(15, 30)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw goals
        for goal in self.goals:
            pos_x, pos_y = goal["pos"]
            center_pixel = (int(pos_x * self.TILE_SIZE + self.TILE_SIZE / 2), int(pos_y * self.TILE_SIZE + self.TILE_SIZE / 2))
            goal_color = tuple(max(0, c - 80) for c in goal["color"])
            pygame.gfxdraw.filled_circle(self.screen, center_pixel[0], center_pixel[1], int(self.TILE_SIZE * 0.35), goal_color)
            pygame.gfxdraw.aacircle(self.screen, center_pixel[0], center_pixel[1], int(self.TILE_SIZE * 0.35), goal_color)

        # Draw walls
        for wall_pos in self.walls:
            rect = pygame.Rect(wall_pos[0] * self.TILE_SIZE, wall_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw push effect
        if self.push_effect:
            life_ratio = self.push_effect["life"] / (self.FPS * 0.2)
            alpha = int(200 * life_ratio)
            color = (255, 255, 255, alpha)
            dx, dy = self.push_effect["dir"]
            
            start_pos = pygame.Vector2(self.CENTER_X, self.CENTER_Y) * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2)
            end_pos = start_pos + pygame.Vector2(dx, dy) * self.TILE_SIZE * (1 - life_ratio) * 5
            
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(temp_surf, color, start_pos, end_pos, int(8 * life_ratio))
            self.screen.blit(temp_surf, (0, 0))

        # Draw blocks
        for block in sorted(self.blocks, key=lambda b: b['pixel_pos'].y):
            rect = pygame.Rect(block["pixel_pos"].x, block["pixel_pos"].y, self.TILE_SIZE, self.TILE_SIZE)
            
            # Shadow
            shadow_rect = rect.copy()
            shadow_rect.y += 4
            shadow_rect.h -= 4
            pygame.draw.rect(self.screen, (0, 0, 0, 50), shadow_rect, border_radius=6)

            # Main block
            border_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, block["color"], border_rect, border_radius=6)
            
            # Highlight
            highlight_color = tuple(min(255, c + 40) for c in block["color"])
            pygame.draw.rect(self.screen, highlight_color, border_rect.inflate(-border_rect.width*0.7, -border_rect.height*0.7).move(2, -2), border_radius=3)

        # Draw particles
        for p in self.particles:
            color = p["color"]
            alpha = int(255 * (p["life"] / 30))
            color_with_alpha = (color[0], color[1], color[2], alpha)
            
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(p["radius"], p["radius"]))

    def _render_ui(self):
        # --- Helper for text rendering ---
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Stage
        draw_text(f"Stage: {self.current_stage}/{self.MAX_STAGES}", self.font_main, self.COLOR_TEXT, (10, 5))
        
        # Timer
        time_left = max(0, self.stage_timer / self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else (255, 100, 100)
        time_text = f"Time: {time_left:.1f}"
        text_width = self.font_main.size(time_text)[0]
        draw_text(time_text, self.font_main, time_color, (self.WIDTH - text_width - 10, 5))

        # Score
        score_text = f"Score: {self.score}"
        text_width = self.font_main.size(score_text)[0]
        draw_text(score_text, self.font_main, self.COLOR_TEXT, ((self.WIDTH - text_width) / 2, self.HEIGHT - 30))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.current_stage > self.MAX_STAGES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_width, text_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, ((self.WIDTH - text_width) / 2, (self.HEIGHT - text_height) / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_left": round(self.stage_timer / self.FPS, 2),
            "blocks_on_goals": sum(1 for b in self.blocks if b["grid_pos"] == self.goals[b["id"]]["pos"])
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pusher Puzzle")
    
    terminated = False
    
    # Game loop
    while not terminated:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Convert observation back to a surface for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()