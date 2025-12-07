import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:37.427511
# Source Brief: brief_00142.md
# Brief Index: 142
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Branch:
    """A helper class to represent a single branch of the fractal tree."""
    def __init__(self, start, angle, length, depth, branch_id):
        self.start = np.array(start, dtype=float)
        self.angle = angle
        self.length = length
        self.depth = depth
        self.id = branch_id
        self.update_geometry()

    def update_geometry(self):
        """Recalculates end point, vector, and normal from angle and length."""
        self.end = self.start + self.length * np.array([math.cos(self.angle), math.sin(self.angle)])
        self.vec = self.end - self.start
        norm_val = np.linalg.norm(self.vec)
        self.vec_normalized = self.vec / norm_val if norm_val > 0 else np.array([0,0])
        self.normal = np.array([-self.vec_normalized[1], self.vec_normalized[0]])

    def rotate(self, rad):
        """Rotates the branch by a given angle in radians."""
        self.angle += rad
        self.update_geometry()

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a falling particle through a rotating fractal tree to collect glowing orbs against the clock."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a branch. Use ↑↓ arrow keys to rotate the selected branch and guide the particle."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and clock
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.render_mode = render_mode

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
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 64)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans", 28)
            self.font_game_over = pygame.font.SysFont("sans", 64)

        # Colors
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_BRANCH = (180, 180, 200)
        self.COLOR_BRANCH_SELECTED = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 255, 0)
        self.COLOR_ORB = (0, 200, 255)
        self.COLOR_TEXT = (220, 220, 220)

        # Game constants
        self.GRAVITY = 0.04
        self.PARTICLE_RADIUS = 7
        self.ORB_BASE_RADIUS = 12
        self.WIN_SCORE = 50
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.BRANCH_ROTATION_SPEED = math.radians(5)
        self.TREE_BASE_Y = self.HEIGHT - 20
        self.WIN_ZONE = pygame.Rect(0, self.TREE_BASE_Y, self.WIDTH, 20)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particle = {}
        self.branches = []
        self.orbs = []
        self.selected_branch_idx = 0
        self.level_depth = 0
        self.branch_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.particle = {
            'pos': np.array([self.WIDTH / 2, 40.0]),
            'vel': np.array([0.0, 0.0]),
            'stuck_frames': 0,
            'trail': []
        }

        self.level_depth = 3
        self._generate_full_game_state()

        return self._get_observation(), self._get_info()

    def _generate_full_game_state(self):
        """Generates the tree and orbs for the current level."""
        self.branches.clear()
        self.orbs.clear()
        self.branch_counter = 0
        
        root_branch = Branch(
            start=(self.WIDTH / 2, self.TREE_BASE_Y),
            angle=-math.pi / 2,
            length=60,
            depth=self.level_depth,
            branch_id=self.branch_counter
        )
        self.branches.append(root_branch)
        self.branch_counter += 1
        self._generate_recursive_branches(root_branch)

        # Sort branches by y-position, then x. Ensures consistent selection order.
        self.branches.sort(key=lambda b: (b.start[1], b.start[0]))
        self.selected_branch_idx = 0 if self.branches else -1
        
        self._spawn_orbs(5)

    def _generate_recursive_branches(self, parent_branch):
        if parent_branch.depth <= 1:
            return

        current_length = parent_branch.length * 0.8
        current_depth = parent_branch.depth - 1
        
        # Left child
        left_angle = parent_branch.angle - math.radians(self.np_random.uniform(20, 45))
        left_branch = Branch(parent_branch.end, left_angle, current_length, current_depth, self.branch_counter)
        self.branches.append(left_branch)
        self.branch_counter += 1
        self._generate_recursive_branches(left_branch)

        # Right child
        right_angle = parent_branch.angle + math.radians(self.np_random.uniform(20, 45))
        right_branch = Branch(parent_branch.end, right_angle, current_length, current_depth, self.branch_counter)
        self.branches.append(right_branch)
        self.branch_counter += 1
        self._generate_recursive_branches(right_branch)

    def _spawn_orbs(self, count):
        leaf_branches = [b for b in self.branches if b.depth == 1]
        if not leaf_branches:
            leaf_branches = self.branches # Failsafe

        for _ in range(count):
            branch = self.np_random.choice(leaf_branches)
            # Spawn orb slightly offset from the branch end
            offset = self.np_random.uniform(5, 25)
            angle = branch.angle + self.np_random.uniform(-math.pi/4, math.pi/4)
            pos = branch.end + offset * np.array([math.cos(angle), math.sin(angle)])
            
            pos[0] = np.clip(pos[0], self.ORB_BASE_RADIUS, self.WIDTH - self.ORB_BASE_RADIUS)
            pos[1] = np.clip(pos[1], self.ORB_BASE_RADIUS, self.HEIGHT - self.ORB_BASE_RADIUS)
            
            self.orbs.append({'pos': pos, 'spawn_step': self.steps})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement, _, _ = action # space_held and shift_held are unused
        
        # 1. Handle player input
        self._handle_action(movement)

        # 2. Update particle physics
        old_y = self.particle['pos'][1]
        self._update_particle()
        
        # Reward for downward progress
        delta_y = self.particle['pos'][1] - old_y
        if delta_y > 0:
            reward += 0.01 * delta_y # Scaled downward reward

        # 3. Handle orb collection
        reward += self._handle_orb_collection()
        
        # 4. Check for level up
        if self.score > 0 and self.score % 10 == (self.level_depth - 3) * 10 and self.level_depth < 8:
            self.level_depth += 1
            self._generate_full_game_state()
            # sfx: level_up_sound

        # 5. Anti-softlock mechanisms
        self._handle_softlocks()

        # 6. Update step counter and check for termination
        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated

        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action(self, movement):
        if not self.branches or self.selected_branch_idx == -1:
            return

        if movement == 1: # Up -> Rotate CCW
            self.branches[self.selected_branch_idx].rotate(-self.BRANCH_ROTATION_SPEED)
        elif movement == 2: # Down -> Rotate CW
            self.branches[self.selected_branch_idx].rotate(self.BRANCH_ROTATION_SPEED)
        elif movement == 3: # Left -> Prev branch
            self.selected_branch_idx = (self.selected_branch_idx - 1) % len(self.branches)
        elif movement == 4: # Right -> Next branch
            self.selected_branch_idx = (self.selected_branch_idx + 1) % len(self.branches)

    def _update_particle(self):
        # Apply gravity
        self.particle['vel'][1] += self.GRAVITY

        # Update position
        self.particle['pos'] += self.particle['vel']

        # Update trail
        self.particle['trail'].append(tuple(self.particle['pos']))
        if len(self.particle['trail']) > 15:
            self.particle['trail'].pop(0)

        # Collisions with branches
        for branch in self.branches:
            p = self.particle['pos']
            a = branch.start
            b = branch.end
            
            ap = p - a
            ab = b - a
            
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
            t = np.clip(t, 0, 1)
            
            closest_point = a + t * ab
            dist_vec = p - closest_point
            distance = np.linalg.norm(dist_vec)

            if distance < self.PARTICLE_RADIUS:
                # Collision detected
                # sfx: bounce_sound
                overlap = self.PARTICLE_RADIUS - distance
                p += (dist_vec / (distance + 1e-6)) * overlap
                
                # Elastic collision response
                normal = dist_vec / (distance + 1e-6)
                vel_component = np.dot(self.particle['vel'], normal)
                self.particle['vel'] -= 1.8 * vel_component * normal # 1.8 for bounciness

        # Boundary collisions
        if self.particle['pos'][0] < self.PARTICLE_RADIUS or self.particle['pos'][0] > self.WIDTH - self.PARTICLE_RADIUS:
            self.particle['vel'][0] *= -0.8
            self.particle['pos'][0] = np.clip(self.particle['pos'][0], self.PARTICLE_RADIUS, self.WIDTH - self.PARTICLE_RADIUS)
        if self.particle['pos'][1] < self.PARTICLE_RADIUS:
            self.particle['vel'][1] *= -0.8
            self.particle['pos'][1] = np.clip(self.particle['pos'][1], self.PARTICLE_RADIUS, self.HEIGHT - self.PARTICLE_RADIUS)


    def _handle_orb_collection(self):
        collected_reward = 0
        orb_radius = self.ORB_BASE_RADIUS * (0.9 ** (self.level_depth - 3))
        
        for orb in self.orbs[:]:
            dist = np.linalg.norm(self.particle['pos'] - orb['pos'])
            if dist < self.PARTICLE_RADIUS + orb_radius:
                self.orbs.remove(orb)
                self.score += 1
                collected_reward += 1.0
                self._spawn_orbs(1) # Spawn a new one
                # sfx: collect_orb_sound
        return collected_reward

    def _handle_softlocks(self):
        # Particle stuck
        if np.linalg.norm(self.particle['vel']) < 0.1 and self.particle['pos'][1] < self.HEIGHT - 50:
            self.particle['stuck_frames'] += 1
        else:
            self.particle['stuck_frames'] = 0

        if self.particle['stuck_frames'] > 180: # 3 seconds
            self.particle['pos'] = np.array([self.WIDTH / 2, 40.0])
            self.particle['vel'] = np.array([0.0, 0.0])
            self.particle['stuck_frames'] = 0
            # sfx: respawn_sound

        # Orbs are too old
        for orb in self.orbs[:]:
            if self.steps - orb['spawn_step'] > 900: # 15 seconds
                self.orbs.remove(orb)
                self._spawn_orbs(1)

    def _check_termination(self):
        # Win condition
        if self.score >= self.WIN_SCORE and self.WIN_ZONE.collidepoint(self.particle['pos']):
            return True, 100.0 # Terminated, Win Reward

        # Lose condition (time out)
        if self.steps >= self.MAX_STEPS:
            return True, -100.0 # Terminated, Lose Penalty

        return False, 0.0

    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level_depth": self.level_depth}

    def _draw_gradient_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render branches
        for i, branch in enumerate(self.branches):
            color = self.COLOR_BRANCH_SELECTED if i == self.selected_branch_idx else self.COLOR_BRANCH
            width = 3 if i == self.selected_branch_idx else 1
            pygame.draw.line(self.screen, color, branch.start, branch.end, width)

        # Render orbs
        orb_radius = self.ORB_BASE_RADIUS * (0.9 ** (self.level_depth - 3))
        for orb in self.orbs:
            pos = orb['pos']
            self._draw_glow_circle(self.screen, self.COLOR_ORB, pos, orb_radius, 15)
            
        # Render particle trail and particle
        if self.particle.get('trail'):
            for i, pos in enumerate(self.particle['trail']):
                alpha = int(255 * (i / len(self.particle['trail'])))
                color = (*self.COLOR_PARTICLE, alpha)
                temp_surf = pygame.Surface((self.PARTICLE_RADIUS*2, self.PARTICLE_RADIUS*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.PARTICLE_RADIUS, self.PARTICLE_RADIUS), int(self.PARTICLE_RADIUS * (i / len(self.particle['trail']))))
                self.screen.blit(temp_surf, (int(pos[0] - self.PARTICLE_RADIUS), int(pos[1] - self.PARTICLE_RADIUS)))

        if self.particle.get('pos') is not None:
            self._draw_glow_circle(self.screen, self.COLOR_PARTICLE, self.particle['pos'], self.PARTICLE_RADIUS, 20)

    def _draw_glow_circle(self, surface, color, pos, radius, max_glow):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(max_glow, 0, -2):
            alpha = int(100 * (1 - i / max_glow))
            glow_color = (*color, alpha)
            temp_surf = pygame.Surface((radius*2 + i*2, radius*2 + i*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (radius + i, radius + i), radius + i)
            surface.blit(temp_surf, (pos_int[0] - radius - i, pos_int[1] - radius - i))
        
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)

    def _render_ui(self):
        # Score
        score_text = f"Orbs: {self.score} / {self.WIN_SCORE}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = f"Time: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            won = self.score >= self.WIN_SCORE
            message = "VICTORY" if won else "TIME UP"
            color = (100, 255, 100) if won else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(message, True, color)
            pos_x = self.WIDTH / 2 - game_over_surf.get_width() / 2
            pos_y = self.HEIGHT / 2 - game_over_surf.get_height() / 2
            self.screen.blit(game_over_surf, (pos_x, pos_y))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Ensure you have pygame installed: pip install pygame
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal Fall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}")
            # Wait for 'R' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False
                clock.tick(env.FPS)

        clock.tick(env.FPS)
        
    env.close()