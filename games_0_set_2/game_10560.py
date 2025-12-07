import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:35:13.727769
# Source Brief: brief_00560.md
# Brief Index: 560
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Quark Arena: A fast-paced, physics-based sports game where players launch
    color-coded quarks into the opponent's nucleus to score points.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A fast-paced, physics-based sports game where players launch quarks into the opponent's nucleus to score points."
    )
    user_guide = (
        "Use ←→ arrow keys to aim your launcher. Hold shift to increase launch power. Press space to fire a quark."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Arena Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.ARENA_CENTER = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.ARENA_RADIUS = 180
        
        # Game Parameters
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS
        self.SCORE_LIMIT = 10
        self.QUARK_RADIUS = 7
        self.NUCLEUS_RADIUS = 35
        self.QUARK_SPEED_MIN = 2
        self.QUARK_SPEED_MAX = 12
        self.PHYSICS_DRAG = 0.01
        self.PLAYER_AIM_RATE = 0.05  # Radians per step
        self.PLAYER_POWER_RATE = 2

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_ARENA_LINE = (30, 40, 70)
        self.COLOR_ARENA_WALL = (60, 80, 140)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_NUCLEUS = (0, 70, 100)
        self.COLOR_OPPONENT = (255, 50, 100)
        self.COLOR_OPPONENT_NUCLEUS = (100, 20, 40)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TRAIL = (150, 150, 180)
        
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
        self.font_ui = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 40, bold=True)
        
        # Initialize state variables
        self.player_nucleus_pos = None
        self.opponent_nucleus_pos = None
        self.player_launch_pos = None
        self.opponent_launch_pos = None
        self.steps = 0
        self.score = 0
        self.player_score = 0
        self.opponent_score = 0
        self.game_over = False
        self.game_outcome = ""
        self.quarks = []
        self.particles = []
        self.player_aim_angle = 0
        self.player_launch_power = 0
        self.player_can_launch = True
        self.player_launch_cooldown = 0
        self.opponent_accuracy = 0
        self.opponent_launch_cooldown = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.player_score = 0
        self.opponent_score = 0
        self.game_over = False
        self.game_outcome = ""
        
        self.quarks = []
        self.particles = []

        # Player state
        self.player_nucleus_pos = pygame.math.Vector2(self.ARENA_CENTER.x, self.ARENA_CENTER.y + self.ARENA_RADIUS + 20)
        self.player_launch_pos = pygame.math.Vector2(self.ARENA_CENTER.x, self.ARENA_CENTER.y + 140)
        self.player_aim_angle = -math.pi / 2
        self.player_launch_power = 50
        self.player_can_launch = True
        self.player_launch_cooldown = 0
        
        # Opponent state
        self.opponent_nucleus_pos = pygame.math.Vector2(self.ARENA_CENTER.x, self.ARENA_CENTER.y - self.ARENA_RADIUS - 20)
        self.opponent_launch_pos = pygame.math.Vector2(self.ARENA_CENTER.x, self.ARENA_CENTER.y - 140)
        self.opponent_accuracy = 0.5
        self.opponent_launch_cooldown = self.np_random.integers(30, 60)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        
        # 1. Handle Player Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_action(movement, space_held, shift_held)
        
        # 2. Update Opponent AI
        self._update_opponent_ai()
        
        # 3. Update Physics and Game Logic
        physics_reward = self._update_physics()
        reward += physics_reward
        
        # 4. Update Visual Effects
        self._update_particles()
        
        # 5. Update Game State
        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward
        
        truncated = False # Truncated is always false for this env
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_action(self, movement, space_held, shift_held):
        # Update aim angle
        if movement == 1: # Up (not used for angle)
            pass
        if movement == 2: # Down (not used for angle)
            pass
        if movement == 3: # Left
            self.player_aim_angle -= self.PLAYER_AIM_RATE
        if movement == 4: # Right
            self.player_aim_angle += self.PLAYER_AIM_RATE
        
        # Normalize angle
        self.player_aim_angle %= (2 * math.pi)

        # Update power
        if shift_held:
            self.player_launch_power += self.PLAYER_POWER_RATE
        elif movement == 0: # No-op decreases power
            self.player_launch_power -= self.PLAYER_POWER_RATE
        self.player_launch_power = np.clip(self.player_launch_power, 0, 100)
        
        # Cooldown management
        if self.player_launch_cooldown > 0:
            self.player_launch_cooldown -= 1
            self.player_can_launch = self.player_launch_cooldown <= 0

        # Launch quark
        if space_held and self.player_can_launch:
            power_ratio = self.player_launch_power / 100.0
            speed = self.QUARK_SPEED_MIN + (self.QUARK_SPEED_MAX - self.QUARK_SPEED_MIN) * power_ratio
            velocity = pygame.math.Vector2(math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)) * speed
            
            self._spawn_quark(self.player_launch_pos, velocity, 'player')
            self._spawn_particles(self.player_launch_pos, self.COLOR_PLAYER, 15, speed) # sfx: player_launch.wav
            self.player_can_launch = False
            self.player_launch_cooldown = 15 # 0.5 second cooldown

    def _update_opponent_ai(self):
        self.opponent_launch_cooldown -= 1
        self.opponent_accuracy = min(1.0, 0.5 + (self.steps / self.MAX_STEPS) * 0.5)

        if self.opponent_launch_cooldown <= 0:
            inaccuracy_offset = (1 - self.opponent_accuracy) * self.NUCLEUS_RADIUS * 2
            target_pos = self.player_nucleus_pos + pygame.math.Vector2(
                self.np_random.uniform(-inaccuracy_offset, inaccuracy_offset),
                self.np_random.uniform(-inaccuracy_offset, inaccuracy_offset)
            )
            
            direction = (target_pos - self.opponent_launch_pos).normalize()
            power = self.np_random.uniform(0.6, 1.0)
            speed = self.QUARK_SPEED_MIN + (self.QUARK_SPEED_MAX - self.QUARK_SPEED_MIN) * power
            velocity = direction * speed
            
            self._spawn_quark(self.opponent_launch_pos, velocity, 'opponent')
            self._spawn_particles(self.opponent_launch_pos, self.COLOR_OPPONENT, 15, speed) # sfx: opponent_launch.wav
            self.opponent_launch_cooldown = self.np_random.integers(45, 90)

    def _update_physics(self):
        reward = 0
        
        # Move quarks and handle wall collisions
        for q in self.quarks:
            q['vel'] *= (1 - self.PHYSICS_DRAG)
            q['pos'] += q['vel']
            q['trail'].append(q['pos'].copy())
            if len(q['trail']) > 15:
                q['trail'].pop(0)

            dist_to_center = q['pos'].distance_to(self.ARENA_CENTER)
            if dist_to_center > self.ARENA_RADIUS - self.QUARK_RADIUS:
                # sfx: wall_bounce.wav
                self._spawn_particles(q['pos'], self.COLOR_ARENA_WALL, 5, q['vel'].length() / 2)
                normal = (q['pos'] - self.ARENA_CENTER).normalize()
                q['vel'] = q['vel'].reflect(normal) * 0.9
                # Push quark back inside to prevent sticking
                q['pos'] = self.ARENA_CENTER + normal * (self.ARENA_RADIUS - self.QUARK_RADIUS)

        # Handle quark-quark collisions
        for i in range(len(self.quarks)):
            for j in range(i + 1, len(self.quarks)):
                q1, q2 = self.quarks[i], self.quarks[j]
                dist = q1['pos'].distance_to(q2['pos'])
                if dist < self.QUARK_RADIUS * 2:
                    if q1['owner'] != q2['owner']:
                        # Neutralization
                        # sfx: neutralization.wav
                        self._spawn_particles((q1['pos'] + q2['pos']) / 2, (255, 255, 255), 30, 5)
                        q1['alive'] = False
                        q2['alive'] = False
                    else:
                        # Bounce
                        # sfx: quark_clack.wav
                        self._resolve_elastic_collision(q1, q2)
                        if q1['owner'] == 'player':
                            reward += 0.5 # Reward for skillful friendly interactions

        # Handle nucleus collisions (scoring)
        for q in self.quarks:
            if not q['alive']: continue
            
            is_player_quark = q['owner'] == 'player'
            target_nucleus_pos = self.opponent_nucleus_pos if is_player_quark else self.player_nucleus_pos
            
            if q['pos'].distance_to(target_nucleus_pos) < self.NUCLEUS_RADIUS + self.QUARK_RADIUS:
                # sfx: score.wav
                if is_player_quark:
                    self.player_score += 1
                    reward += 1
                    self._spawn_particles(q['pos'], self.COLOR_PLAYER, 50, 8)
                else:
                    self.opponent_score += 1
                    reward -= 1
                    self._spawn_particles(q['pos'], self.COLOR_OPPONENT, 50, 8)
                q['alive'] = False
        
        # Clean up dead quarks
        new_quarks = []
        for q in self.quarks:
            if q['alive']:
                new_quarks.append(q)
            else:
                # Reward for quark's final position
                if q['owner'] == 'player':
                    if q['pos'].y < self.ARENA_CENTER.y: # Opponent's side
                        reward += 0.1
                    else: # Own side
                        reward -= 0.1
        self.quarks = new_quarks
        
        return reward

    def _resolve_elastic_collision(self, q1, q2):
        v1, v2 = q1['vel'], q2['vel']
        p1, p2 = q1['pos'], q2['pos']
        
        collision_normal = (p2 - p1).normalize()
        relative_velocity = v1 - v2
        speed = relative_velocity.dot(collision_normal)

        if speed > 0: return # Moving away from each other
        
        impulse = (2 * speed) / 2 # Assuming equal mass of 1
        q1['vel'] -= impulse * collision_normal
        q2['vel'] += impulse * collision_normal
        
        # Separate overlapping circles
        overlap = self.QUARK_RADIUS * 2 - p1.distance_to(p2)
        if overlap > 0:
            p1 -= collision_normal * (overlap / 2)
            p2 += collision_normal * (overlap / 2)


    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['vel'] *= 0.95 # Particle drag
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _check_termination(self):
        terminated = False
        reward = 0
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if self.player_score > self.opponent_score:
                reward = 10
                self.game_outcome = "YOU WIN!"
            elif self.opponent_score > self.player_score:
                reward = -10
                self.game_outcome = "YOU LOSE"
            else:
                self.game_outcome = "DRAW"
        elif self.player_score >= self.SCORE_LIMIT:
            terminated = True
            reward = 10
            self.game_outcome = "YOU WIN!"
        elif self.opponent_score >= self.SCORE_LIMIT:
            terminated = True
            reward = -10
            self.game_outcome = "YOU LOSE"
        
        if terminated:
            self.game_over = True
            
        return terminated, reward

    def _spawn_quark(self, pos, vel, owner):
        color = self.COLOR_PLAYER if owner == 'player' else self.COLOR_OPPONENT
        self.quarks.append({
            'pos': pos.copy(), 'vel': vel.copy(), 'color': color,
            'owner': owner, 'alive': True, 'trail': []
        })

    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(max_speed * 0.2, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color,
                'lifetime': self.np_random.integers(10, 25),
                'size': self.np_random.uniform(1, 4)
            })
            
    def _draw_glow_circle(self, surface, color, pos, radius, glow_strength=5):
        glow_color = pygame.Color(color)
        for i in range(glow_strength, 0, -1):
            alpha = int(100 * (1 - i / glow_strength))
            pygame.gfxdraw.filled_circle(
                surface, int(pos.x), int(pos.y),
                int(radius + i * 2),
                (glow_color.r, glow_color.g, glow_color.b, alpha)
            )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena
        pygame.gfxdraw.filled_circle(self.screen, int(self.ARENA_CENTER.x), int(self.ARENA_CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA_LINE)
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER.x), int(self.ARENA_CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA_WALL)
        pygame.draw.line(self.screen, self.COLOR_ARENA_WALL, (self.ARENA_CENTER.x - self.ARENA_RADIUS, self.ARENA_CENTER.y), (self.ARENA_CENTER.x + self.ARENA_RADIUS, self.ARENA_CENTER.y), 2)

        # Draw nuclei
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, self.player_nucleus_pos, self.NUCLEUS_RADIUS)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_nucleus_pos.x), int(self.player_nucleus_pos.y), self.NUCLEUS_RADIUS, self.COLOR_PLAYER_NUCLEUS)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_nucleus_pos.x), int(self.player_nucleus_pos.y), self.NUCLEUS_RADIUS, self.COLOR_PLAYER)
        
        self._draw_glow_circle(self.screen, self.COLOR_OPPONENT, self.opponent_nucleus_pos, self.NUCLEUS_RADIUS)
        pygame.gfxdraw.filled_circle(self.screen, int(self.opponent_nucleus_pos.x), int(self.opponent_nucleus_pos.y), self.NUCLEUS_RADIUS, self.COLOR_OPPONENT_NUCLEUS)
        pygame.gfxdraw.aacircle(self.screen, int(self.opponent_nucleus_pos.x), int(self.opponent_nucleus_pos.y), self.NUCLEUS_RADIUS, self.COLOR_OPPONENT)

        # Draw particles
        for p in self.particles:
            alpha = p['lifetime'] * 10
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)

        # Draw quarks and trails
        for q in self.quarks:
            # Trail
            if q.get('trail'):
                for i, t_pos in enumerate(q['trail']):
                    alpha = int(255 * (i / len(q['trail']))) * 0.5
                    color = (*q['color'], alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(t_pos.x), int(t_pos.y), int(self.QUARK_RADIUS * (i / len(q['trail']))), color)
            # Quark
            self._draw_glow_circle(self.screen, q['color'], q['pos'], self.QUARK_RADIUS)
            pygame.gfxdraw.filled_circle(self.screen, int(q['pos'].x), int(q['pos'].y), self.QUARK_RADIUS, q['color'])
            pygame.gfxdraw.aacircle(self.screen, int(q['pos'].x), int(q['pos'].y), self.QUARK_RADIUS, (255,255,255))
        
        # Draw player aim indicator
        if self.player_can_launch:
            # Trajectory
            power_ratio = self.player_launch_power / 100.0
            speed = self.QUARK_SPEED_MIN + (self.QUARK_SPEED_MAX - self.QUARK_SPEED_MIN) * power_ratio
            sim_pos = self.player_launch_pos.copy()
            sim_vel = pygame.math.Vector2(math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)) * speed
            for i in range(20):
                sim_vel *= (1 - self.PHYSICS_DRAG)
                sim_pos += sim_vel
                if i % 2 == 0:
                    alpha = 200 - i * 8
                    pygame.gfxdraw.filled_circle(self.screen, int(sim_pos.x), int(sim_pos.y), 2, (*self.COLOR_PLAYER, alpha))

    def _render_ui(self):
        # Score
        player_score_text = self.font_ui.render(f"{self.player_score}", True, self.COLOR_PLAYER)
        self.screen.blit(player_score_text, (20, self.SCREEN_HEIGHT - 30))
        
        opponent_score_text = self.font_ui.render(f"{self.opponent_score}", True, self.COLOR_OPPONENT)
        self.screen.blit(opponent_score_text, (20, 10))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps) // self.metadata['render_fps']
        timer_text = self.font_ui.render(f"{time_left}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(center=(self.SCREEN_WIDTH / 2, 20))
        self.screen.blit(timer_text, text_rect)
        
        # Power meter
        power_bar_width = 100
        power_bar_height = 10
        power_bar_x = self.SCREEN_WIDTH - power_bar_width - 20
        power_bar_y = self.SCREEN_HEIGHT - power_bar_height - 15
        
        pygame.draw.rect(self.screen, self.COLOR_ARENA_LINE, (power_bar_x, power_bar_y, power_bar_width, power_bar_height))
        current_power_width = int(power_bar_width * (self.player_launch_power / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (power_bar_x, power_bar_y, current_power_width, power_bar_height))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(overlay, (0,0))
            
            msg_text = self.font_msg.render(self.game_outcome, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block is for manual play and visualization.
    # It will not be executed in the headless evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Quark Arena")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Main game loop
    while not done:
        # Action defaults
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    env.close()