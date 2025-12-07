import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:01:21.125848
# Source Brief: brief_00763.md
# Brief Index: 763
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

# Helper classes for game objects
class Block:
    def __init__(self, x, y, mass, is_static, color, building_id):
        self.pos = pygame.Vector2(x, y)
        self.old_pos = pygame.Vector2(x, y)
        self.acc = pygame.Vector2(0, 0)
        self.mass = mass
        self.radius = 8
        self.is_static = is_static
        self.color = color
        self.building_id = building_id

class Constraint:
    def __init__(self, block_a_idx, block_b_idx, length):
        self.block_a_idx = block_a_idx
        self.block_b_idx = block_b_idx
        self.length = length

class Particle:
    def __init__(self, x, y, vx, vy, lifespan, color, radius):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.radius = radius

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Use tools like Push and Explode to demolish target structures while avoiding collateral damage."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to use the selected tool and shift to cycle between tools."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 18, 42)
    COLOR_GROUND = (60, 60, 80)
    COLOR_TARGET = (255, 80, 80)
    COLOR_NONTARGET = (80, 120, 255)
    COLOR_TOOL = (50, 255, 150)
    COLOR_EXPLOSION = (255, 200, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    # Screen
    WIDTH, HEIGHT = 640, 400
    
    # Physics
    GRAVITY = 0.1
    GROUND_Y = 350
    PHYSICS_SUBSTEPS = 8
    DEMOLITION_Y_THRESHOLD = 345

    # Gameplay
    MAX_STEPS = 2000
    CURSOR_SPEED = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces:
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_tool = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        # Game state variables - initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.blocks = []
        self.constraints = []
        self.particles = []
        self.buildings = {}
        self.demolished_buildings = set()
        self.level = 1

        self.tools = [
            {"name": "PUSH", "cost": 1, "color": (50, 255, 150)},
            {"name": "EXPLODE", "cost": 5, "color": (255, 150, 50)}
        ]
        self.selected_tool_idx = 0

        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self.blocks = []
        self.constraints = []
        self.particles = []
        self.buildings = {}
        self.demolished_buildings = set()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Input Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        step_reward = 0

        # Cycle tool on SHIFT press
        if shift_held and not self.prev_shift_held:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.tools)
            # sfx: tool_switch.wav

        # Use tool on SPACE press
        if space_held and not self.prev_space_held:
            tool_cost = self.tools[self.selected_tool_idx]["cost"]
            self.score -= tool_cost
            step_reward -= tool_cost
            self._use_tool()
            # sfx: use_tool.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Update ---
        self.steps += 1
        
        initial_constraints = len(self.constraints)
        self._update_physics()
        broken_constraints = initial_constraints - len(self.constraints)
        # For simplicity, we'll give a small reward for any destruction.
        # The main reward comes from building demolition.
        if broken_constraints > 0:
            step_reward += 0.01 * broken_constraints

        self._update_particles()
        
        # Check for demolished buildings and calculate rewards
        step_reward += self._check_building_status()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            # Calculate terminal reward
            all_targets_demolished = True
            for b_id, b_data in self.buildings.items():
                if b_data['type'] == 'target' and b_id not in self.demolished_buildings:
                    all_targets_demolished = False
                    break
            
            if all_targets_demolished:
                step_reward += 50
                self.score += 50
                self.level += 0.1 # Difficulty progression
            else:
                step_reward -= 50
                self.score -= 50
            self.game_over = True

        self.score += step_reward

        # Truncated is always False as there is no condition for it
        truncated = False
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _use_tool(self):
        tool_name = self.tools[self.selected_tool_idx]["name"]
        if tool_name == "PUSH":
            # sfx: push.wav
            for i, block in enumerate(self.blocks):
                if not block.is_static:
                    dist_vec = block.pos - self.cursor_pos
                    dist = dist_vec.length()
                    if dist < 50:
                        force_mag = 10 * (1 - dist / 50)
                        block.acc += dist_vec.normalize() * force_mag if dist > 0 else pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)) * force_mag
        
        elif tool_name == "EXPLODE":
            # sfx: explosion.wav
            radius = 60
            strength = 15
            # Visual effect
            self.particles.append(Particle(self.cursor_pos.x, self.cursor_pos.y, 0, 0, 10, self.COLOR_EXPLOSION, radius))
            for _ in range(50):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 8)
                life = self.np_random.integers(15, 30)
                p_radius = self.np_random.uniform(1, 4)
                self.particles.append(Particle(self.cursor_pos.x, self.cursor_pos.y, math.cos(angle) * speed, math.sin(angle) * speed, life, self.COLOR_EXPLOSION, p_radius))

            # Physics effect
            for i, block in enumerate(self.blocks):
                dist_vec = block.pos - self.cursor_pos
                dist = dist_vec.length()
                if dist < radius and dist > 0:
                    force_mag = strength * (1 - dist / radius)
                    block.acc += dist_vec.normalize() * force_mag
                    # Weaken constraints
                    self.constraints = [c for c in self.constraints if not (c.block_a_idx == i or c.block_b_idx == i) or self.np_random.random() > 0.6]

    def _update_physics(self):
        dt = 1.0 / self.PHYSICS_SUBSTEPS
        for _ in range(self.PHYSICS_SUBSTEPS):
            # Apply forces
            for block in self.blocks:
                if not block.is_static:
                    block.acc += pygame.Vector2(0, self.GRAVITY)
            
            # Verlet integration
            for block in self.blocks:
                if not block.is_static:
                    vel = block.pos - block.old_pos
                    block.old_pos = block.pos
                    block.pos = block.pos + vel + block.acc * dt * dt
                    block.acc = pygame.Vector2(0, 0)
            
            # Solve constraints and collisions
            broken_constraints_indices = set()
            for i, c in enumerate(self.constraints):
                p1 = self.blocks[c.block_a_idx]
                p2 = self.blocks[c.block_b_idx]
                axis = p1.pos - p2.pos
                dist = axis.length()
                if dist == 0: continue
                
                diff = (dist - c.length) / dist
                
                # Break constraint if overstretched
                if abs(dist - c.length) > c.length * 0.8:
                    broken_constraints_indices.add(i)
                    continue

                if p1.is_static:
                    p2.pos += axis * diff
                elif p2.is_static:
                    p1.pos -= axis * diff
                else:
                    p1.pos -= axis * 0.5 * diff
                    p2.pos += axis * 0.5 * diff
            
            if broken_constraints_indices:
                # sfx: crack.wav
                for i in sorted(list(broken_constraints_indices), reverse=True):
                    c = self.constraints[i]
                    p1 = self.blocks[c.block_a_idx]
                    p2 = self.blocks[c.block_b_idx]
                    mid_point = (p1.pos + p2.pos) / 2
                    for _ in range(3):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(0.5, 2)
                        life = self.np_random.integers(10, 20)
                        self.particles.append(Particle(mid_point.x, mid_point.y, math.cos(angle)*speed, math.sin(angle)*speed, life, (120,120,120), 2))
                    del self.constraints[i]

            # Ground collision
            for block in self.blocks:
                if not block.is_static and block.pos.y > self.GROUND_Y - block.radius:
                    vel = block.pos - block.old_pos
                    block.pos.y = self.GROUND_Y - block.radius
                    block.old_pos.y = block.pos.y + vel.y * 0.3 # Bounciness
                    block.old_pos.x = block.pos.x - vel.x * 0.95 # Friction

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.pos += p.vel
            p.lifespan -= 1
            if p.color == self.COLOR_EXPLOSION and p.max_lifespan > 10: # Is an explosion shockwave
                p.radius *= 0.95

    def _check_building_status(self):
        reward = 0
        active_blocks_by_building = {b_id: [] for b_id in self.buildings}
        
        # Find which blocks are still "active" (above ground)
        for i, block in enumerate(self.blocks):
            if block.pos.y < self.DEMOLITION_Y_THRESHOLD:
                if block.building_id in active_blocks_by_building:
                    active_blocks_by_building[block.building_id].append(i)
        
        # Check for newly demolished buildings
        for b_id, b_data in self.buildings.items():
            if b_id not in self.demolished_buildings and len(active_blocks_by_building[b_id]) == 0:
                self.demolished_buildings.add(b_id)
                # sfx: building_collapse.wav
                if b_data['type'] == 'target':
                    reward += 10
                else: # Non-target
                    reward -= 20
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        
        all_targets_demolished = True
        for b_id, b_data in self.buildings.items():
            if b_data['type'] == 'target' and b_id not in self.demolished_buildings:
                all_targets_demolished = False
                break
        
        return all_targets_demolished

    def _generate_level(self):
        num_buildings = self.np_random.integers(2, 5)
        num_targets = max(1, int(num_buildings / 2))
        
        building_types = ['target'] * num_targets + ['nontarget'] * (num_buildings - num_targets)
        self.np_random.shuffle(building_types)

        x_positions = np.linspace(100, self.WIDTH - 100, num_buildings)
        self.np_random.shuffle(x_positions)

        for i in range(num_buildings):
            b_id = i
            b_type = building_types[i]
            color = self.COLOR_TARGET if b_type == 'target' else self.COLOR_NONTARGET
            
            width = self.np_random.integers(2, 4)
            height = self.np_random.integers(3, 6 + int(self.level))
            start_x = x_positions[i]

            self.buildings[b_id] = {'type': b_type, 'block_indices': set()}
            
            block_grid = {}
            for y in range(height):
                for x in range(width):
                    is_static = (y == 0)
                    block_idx = len(self.blocks)
                    px = start_x + x * 17
                    py = self.GROUND_Y - y * 17
                    self.blocks.append(Block(px, py, 1.0, is_static, color, b_id))
                    self.buildings[b_id]['block_indices'].add(block_idx)
                    block_grid[(x,y)] = block_idx
            
            # Add constraints
            for y in range(height):
                for x in range(width):
                    idx1 = block_grid[(x,y)]
                    if x + 1 < width:
                        idx2 = block_grid[(x+1, y)]
                        dist = (self.blocks[idx1].pos - self.blocks[idx2].pos).length()
                        self.constraints.append(Constraint(idx1, idx2, dist))
                    if y + 1 < height:
                        idx2 = block_grid[(x, y+1)]
                        dist = (self.blocks[idx1].pos - self.blocks[idx2].pos).length()
                        self.constraints.append(Constraint(idx1, idx2, dist))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Constraints
        for c in self.constraints:
            p1 = self.blocks[c.block_a_idx].pos
            p2 = self.blocks[c.block_b_idx].pos
            pygame.draw.aaline(self.screen, (80, 80, 100), p1, p2)
            
        # Blocks
        for block in self.blocks:
            color = block.color
            if block.is_static:
                color = tuple(c*0.6 for c in color)
            pygame.gfxdraw.filled_circle(self.screen, int(block.pos.x), int(block.pos.y), int(block.radius), color)
            pygame.gfxdraw.aacircle(self.screen, int(block.pos.x), int(block.pos.y), int(block.radius), tuple(c*0.8 for c in color))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            if p.color == self.COLOR_EXPLOSION and p.max_lifespan > 10: # Shockwave
                if p.radius > 1:
                    pygame.gfxdraw.aacircle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), (*p.color, alpha))
            else: # Debris
                pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), (*p.color, alpha))

        # Cursor
        cursor_color = self.tools[self.selected_tool_idx]['color']
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 8, (*cursor_color, 100))
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 8, cursor_color)
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos.x - 5, self.cursor_pos.y), (self.cursor_pos.x + 5, self.cursor_pos.y))
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos.x, self.cursor_pos.y - 5), (self.cursor_pos.x, self.cursor_pos.y + 5))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0]+1, pos[1]+1))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, pos)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (self.WIDTH - 150, 10))
        
        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_color = self.COLOR_TEXT if time_left > 200 else (255, 100, 100)
        time_text = f"TIME: {time_left}"
        draw_text(time_text, self.font_ui, time_color, (10, 10))

        # Selected Tool
        tool = self.tools[self.selected_tool_idx]
        tool_text = f"TOOL: {tool['name']} (Cost: {tool['cost']})"
        text_width = self.font_tool.size(tool_text)[0]
        draw_text(tool_text, self.font_tool, tool['color'], (self.WIDTH/2 - text_width/2, self.HEIGHT - 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "targets_remaining": sum(1 for b_id, b in self.buildings.items() if b['type'] == 'target' and b_id not in self.demolished_buildings)
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Demolition Master")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n--- Controls ---")
    print("Arrows: Move cursor")
    print("Space: Use tool")
    print("Shift: Cycle tool")
    print("Q: Quit")
    print("R: Reset")
    
    while not done:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()