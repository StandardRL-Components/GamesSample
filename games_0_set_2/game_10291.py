import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:14:58.725012
# Source Brief: brief_00291.md
# Brief Index: 291
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls three robotic arms with delayed actions
    to assemble a circuit board within a time limit.

    **Objective:** Place all 10 components in their correct slots to complete the circuit.
    **Challenge:** Actions for each arm are queued and executed after a 1-second delay,
    requiring foresight and planning to avoid collisions and manage time effectively.

    **Action Space (`MultiDiscrete([5, 2, 2])`):**
    - `action[0]` (0-4): Controls Arm 1's movement (None, Up, Down, Left, Right).
    - `action[1]` (0-1): Controls Arm 2 (0: Move Forward, 1: Turn Clockwise).
    - `action[2]` (0-1): Controls Arm 3 (0: Move Forward, 1: Turn Clockwise).

    **State:** The state includes the positions, orientations, and status (idle, moving,
    stunned, holding component) of the three arms, the status of the 10 circuit
    components (at source, held, or placed), and the remaining time.

    **Rewards:**
    - +0.1 for successfully placing a component.
    - -0.1 for a collision between two arms.
    - +105 upon winning (completing the circuit).
    - -100 upon losing (running out of time).

    **Visuals:**
    The environment features a clean, futuristic aesthetic with glowing electrical paths,
    smoothly animated arms, and clear status indicators to provide a polished and
    informative gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three robotic arms with delayed actions to assemble a circuit board. "
        "Place all components in their slots before time runs out, avoiding collisions."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) for the blue arm. "
        "Use W/A for the cyan arm and I/J for the green arm to move forward or turn."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 90
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.ACTION_DELAY_FRAMES = self.FPS  # 1-second delay

        # Visuals & Physics
        self.ARM_RADIUS = 12
        self.ARM_SPEED = 4.0
        self.ARM_GRID_MOVE_SIZE = 40
        self.COMPONENT_SIZE = 10
        self.STUN_DURATION_FRAMES = int(0.5 * self.FPS)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 45, 60)
        self.COLOR_SLOT = (10, 20, 30)
        self.COLOR_ARM_1 = (0, 150, 255)
        self.COLOR_ARM_2 = (50, 200, 200)
        self.COLOR_ARM_3 = (100, 255, 150)
        self.COLOR_STUN = (255, 50, 50)
        self.COLOR_COMPONENT_SOURCE = (180, 180, 200)
        self.COLOR_COMPONENT_PLACED = (100, 255, 100)
        self.COLOR_PATH_ACTIVE = (255, 220, 50)
        self.COLOR_TEXT = (220, 220, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except IOError:
            print("Default font not found, using fallback.")
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)
            
        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left_frames = 0
        self.arms = []
        self.components = []
        self.action_queue = []
        self.last_collisions = set()
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left_frames = self.MAX_STEPS
        self.action_queue = []
        self.last_collisions = set()
        self.particles = []
        
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # Unpack and queue actions with delay
        self._queue_actions(action)

        # Update core game logic for one frame
        reward += self._update_game_state()

        self.steps += 1
        self.time_left_frames -= 1

        terminated = self._check_termination()
        
        if terminated:
            placed_count = sum(1 for c in self.components if c['state'] == 'placed')
            if placed_count == len(self.components):
                # Win condition
                reward += 5.0  # Final component placement bonus
                reward += 100.0 # Win bonus
            else:
                # Loss condition (time out)
                reward -= 100.0
        
        self.score += reward

        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _queue_actions(self, action):
        movement_arm1, action_arm2, action_arm3 = action[0], action[1], action[2]
        execution_frame = self.steps + self.ACTION_DELAY_FRAMES

        # Arm 1: Direct movement command
        if movement_arm1 != 0:
            self.action_queue.append({'frame': execution_frame, 'arm_idx': 0, 'type': 'move', 'value': movement_arm1})
        
        # Arm 2 & 3: Turn or move forward command
        self.action_queue.append({'frame': execution_frame, 'arm_idx': 1, 'type': 'turn_move', 'value': action_arm2})
        self.action_queue.append({'frame': execution_frame, 'arm_idx': 2, 'type': 'turn_move', 'value': action_arm3})

    def _process_action_queue(self):
        due_actions = [a for a in self.action_queue if a['frame'] <= self.steps]
        self.action_queue = [a for a in self.action_queue if a['frame'] > self.steps]

        for act in due_actions:
            arm = self.arms[act['arm_idx']]
            if arm['state'] == 'stunned':
                continue # Action fizzles if stunned

            if act['type'] == 'move':
                # Arm 1 movement
                dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(act['value'], (0, 0))
                arm['target_pos'] += pygame.math.Vector2(dx, dy) * self.ARM_GRID_MOVE_SIZE
            
            elif act['type'] == 'turn_move':
                # Arm 2 & 3 movement
                if act['value'] == 1: # Turn 90 deg clockwise
                    arm['orientation'] = (arm['orientation'] + 1) % 4
                else: # Move forward
                    dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][arm['orientation']]
                    arm['target_pos'] += pygame.math.Vector2(dx, dy) * self.ARM_GRID_MOVE_SIZE

            # Clamp target position to stay within screen bounds
            arm['target_pos'].x = max(self.ARM_RADIUS, min(self.WIDTH - self.ARM_RADIUS, arm['target_pos'].x))
            arm['target_pos'].y = max(self.ARM_RADIUS, min(self.HEIGHT - self.ARM_RADIUS, arm['target_pos'].y))

    def _update_game_state(self):
        step_reward = 0.0
        
        self._process_action_queue()

        # Update arms (stun, movement, interactions)
        for i, arm in enumerate(self.arms):
            if arm['stun_timer'] > 0:
                arm['stun_timer'] -= 1
                if arm['stun_timer'] <= 0:
                    arm['state'] = 'idle'
            
            if arm['state'] != 'stunned':
                dist_to_target = arm['pos'].distance_to(arm['target_pos'])
                if dist_to_target > 1:
                    arm['state'] = 'moving'
                    # Interpolate movement for smoothness
                    arm['pos'] = arm['pos'].lerp(arm['target_pos'], self.ARM_SPEED / self.FPS)
                elif arm['state'] == 'moving':
                    arm['pos'] = pygame.math.Vector2(arm['target_pos']) # Snap to final position
                    arm['state'] = 'idle'
                    step_reward += self._check_interactions(arm)
        
        # Update held component positions
        for arm in self.arms:
            if arm['held_component_idx'] is not None:
                self.components[arm['held_component_idx']]['pos'] = pygame.math.Vector2(arm['pos'])

        # Check for arm-arm collisions
        current_collisions = set()
        for i in range(len(self.arms)):
            for j in range(i + 1, len(self.arms)):
                if self.arms[i]['pos'].distance_to(self.arms[j]['pos']) < self.ARM_RADIUS * 2:
                    current_collisions.add(tuple(sorted((i, j))))
        
        new_collisions = current_collisions - self.last_collisions
        for i, j in new_collisions:
            self.arms[i]['state'] = 'stunned'
            self.arms[i]['stun_timer'] = self.STUN_DURATION_FRAMES
            self.arms[j]['state'] = 'stunned'
            self.arms[j]['stun_timer'] = self.STUN_DURATION_FRAMES
            step_reward -= 0.1 # Collision penalty
            # sfx: collision_zap.wav

        self.last_collisions = current_collisions
        
        self._update_particles()
        return step_reward

    def _check_interactions(self, arm):
        # Check for placing a component
        if arm['held_component_idx'] is not None:
            comp = self.components[arm['held_component_idx']]
            if arm['pos'].distance_to(comp['target_pos']) < 5:
                comp['state'] = 'placed'
                comp['pos'] = comp['target_pos']
                arm['held_component_idx'] = None
                # sfx: component_place.wav
                self._create_particles(comp['pos'], self.COLOR_COMPONENT_PLACED, 20)
                return 0.1 # Reward for placing
        
        # Check for picking up a component
        else:
            for i, comp in enumerate(self.components):
                if comp['state'] == 'source' and arm['pos'].distance_to(comp['source_pos']) < 5:
                    comp['state'] = 'held'
                    arm['held_component_idx'] = i
                    # sfx: component_pickup.wav
                    return 0.0
        return 0.0

    def _check_termination(self):
        if self.game_over:
            return True
        
        placed_count = sum(1 for c in self.components if c['state'] == 'placed')
        if placed_count == len(self.components):
            self.game_over = True
            # sfx: win_jingle.wav
            return True
            
        if self.time_left_frames <= 0:
            self.game_over = True
            # sfx: lose_buzzer.wav
            return True
            
        return False

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
            "time_left": self.time_left_frames / self.FPS,
            "components_placed": sum(1 for c in self.components if c['state'] == 'placed')
        }

    def _setup_level(self):
        self.arms = [
            {'color': self.COLOR_ARM_1, 'pos': pygame.math.Vector2(100, 200), 'target_pos': pygame.math.Vector2(100, 200), 'state': 'idle', 'stun_timer': 0, 'held_component_idx': None, 'orientation': 0},
            {'color': self.COLOR_ARM_2, 'pos': pygame.math.Vector2(320, 100), 'target_pos': pygame.math.Vector2(320, 100), 'state': 'idle', 'stun_timer': 0, 'held_component_idx': None, 'orientation': 0},
            {'color': self.COLOR_ARM_3, 'pos': pygame.math.Vector2(540, 200), 'target_pos': pygame.math.Vector2(540, 200), 'state': 'idle', 'stun_timer': 0, 'held_component_idx': None, 'orientation': 0}
        ]
        
        self.components = []
        source_x_start = 80
        source_y = 360
        path_points = [
            (150, 50), (250, 50), (350, 50), (450, 50), (550, 50),
            (550, 150), (450, 150), (350, 150), (250, 150), (150, 150)
        ]
        for i in range(10):
            source_pos = pygame.math.Vector2(source_x_start + i * 50, source_y)
            target_pos = pygame.math.Vector2(path_points[i])
            self.components.append({
                'id': i, 'source_pos': source_pos, 'target_pos': target_pos,
                'pos': source_pos, 'state': 'source'
            })

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw component slots and sources
        for comp in self.components:
            pygame.draw.rect(self.screen, self.COLOR_SLOT, (*comp['target_pos'] - (self.COMPONENT_SIZE, self.COMPONENT_SIZE), self.COMPONENT_SIZE*2, self.COMPONENT_SIZE*2), border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_GRID, (*comp['source_pos'] - (self.COMPONENT_SIZE, self.COMPONENT_SIZE), self.COMPONENT_SIZE*2, self.COMPONENT_SIZE*2), 2, border_radius=3)

        # Draw active circuit path
        placed_points = [c['pos'] for c in self.components if c['state'] == 'placed']
        if len(placed_points) > 1:
            # Glow effect
            pygame.draw.lines(self.screen, self.COLOR_PATH_ACTIVE, False, placed_points, width=5)
            # Core line
            pygame.draw.lines(self.screen, (255, 255, 255), False, placed_points, width=1)

        # Draw components
        for comp in self.components:
            color = self.COLOR_COMPONENT_SOURCE if comp['state'] != 'placed' else self.COLOR_COMPONENT_PLACED
            pygame.gfxdraw.filled_circle(self.screen, int(comp['pos'].x), int(comp['pos'].y), self.COMPONENT_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int(comp['pos'].x), int(comp['pos'].y), self.COMPONENT_SIZE, color)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))
            
        # Draw arms
        for i, arm in enumerate(self.arms):
            self._draw_arm(self.screen, arm, i)

    def _draw_arm(self, screen, arm, index):
        pos = (int(arm['pos'].x), int(arm['pos'].y))
        
        # Stun effect
        if arm['state'] == 'stunned':
            pulse = abs(math.sin(self.steps * 0.3))
            radius = int(self.ARM_RADIUS * (1.2 + pulse * 0.4))
            alpha = int(100 + pulse * 100)
            pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], radius, (*self.COLOR_STUN, alpha))
            pygame.gfxdraw.aacircle(screen, pos[0], pos[1], radius, (*self.COLOR_STUN, alpha))
        
        # Arm base
        pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], self.ARM_RADIUS, arm['color'])
        pygame.gfxdraw.aacircle(screen, pos[0], pos[1], self.ARM_RADIUS, tuple(min(255, c+50) for c in arm['color']))

        # Orientation pointer (for arms 2 and 3)
        if index > 0:
            angle = arm['orientation'] * math.pi / 2
            p1 = (pos[0] + self.ARM_RADIUS * math.sin(angle), pos[1] - self.ARM_RADIUS * math.cos(angle))
            p2 = (pos[0] + self.ARM_RADIUS * 0.5 * math.sin(angle + math.pi/2), pos[1] - self.ARM_RADIUS * 0.5 * math.cos(angle + math.pi/2))
            p3 = (pos[0] + self.ARM_RADIUS * 0.5 * math.sin(angle - math.pi/2), pos[1] - self.ARM_RADIUS * 0.5 * math.cos(angle - math.pi/2))
            pygame.gfxdraw.aapolygon(screen, (p1, p2, p3), (255, 255, 255))
            pygame.gfxdraw.filled_polygon(screen, (p1, p2, p3), (255, 255, 255))

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {self.time_left_frames / self.FPS:.1f}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Components placed
        placed_count = sum(1 for c in self.components if c['state'] == 'placed')
        comp_str = f"COMPONENTS: {placed_count} / {len(self.components)}"
        comp_surf = self.font_large.render(comp_str, True, self.COLOR_TEXT)
        self.screen.blit(comp_surf, (10, 10))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # We need to re-init pygame to create a display window
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robotic Arm Circuit Assembly")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    running = True
    total_reward = 0
    
    # Store the state of keys to create a MultiDiscrete action
    keys_pressed = {
        'up': False, 'down': False, 'left': False, 'right': False, # Arm 1
        'w': False, 'a': False, # Arm 2
        'i': False, 'j': False, # Arm 3
    }

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------")

    while running:
        # --- Event Handling ---
        action_taken_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r: obs, info = env.reset(); total_reward = 0
                
                # Update key states
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_w, pygame.K_a, pygame.K_i, pygame.K_j]:
                    action_taken_this_frame = True

        # --- Action Assembly ---
        keys = pygame.key.get_pressed()
        arm1_action = 0
        if keys[pygame.K_UP]: arm1_action = 1
        elif keys[pygame.K_DOWN]: arm1_action = 2
        elif keys[pygame.K_LEFT]: arm1_action = 3
        elif keys[pygame.K_RIGHT]: arm1_action = 4

        arm2_action = 0
        if keys[pygame.K_w]: arm2_action = 0
        elif keys[pygame.K_a]: arm2_action = 1

        arm3_action = 0
        if keys[pygame.K_i]: arm3_action = 0
        elif keys[pygame.K_j]: arm3_action = 1
        
        action = [arm1_action, arm2_action, arm3_action]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()