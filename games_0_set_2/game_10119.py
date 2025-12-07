import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:57:40.853658
# Source Brief: brief_00119.md
# Brief Index: 119
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper Classes
class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.98
        self.vel[1] *= 0.98
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface, camera_offset):
        if self.lifetime <= 0:
            return
        
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        current_size = int(self.size * (self.lifetime / self.max_lifetime))
        if current_size <= 0: return

        x, y = int(self.pos[0] - camera_offset[0]), int(self.pos[1] - camera_offset[1])
        
        # Simple circle for performance
        temp_surface = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (*self.color, alpha), (current_size, current_size), current_size)
        surface.blit(temp_surface, (x - current_size, y - current_size))


class Projectile:
    def __init__(self, pos, angle, owner):
        self.pos = list(pos)
        self.angle = angle
        self.owner = owner
        self.speed = 20
        self.vel = [math.cos(self.angle) * self.speed, math.sin(self.angle) * self.speed]
        self.lifetime = 60 # 2 seconds at 30fps
        self.radius = 5

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface, camera_offset):
        x, y = int(self.pos[0] - camera_offset[0]), int(self.pos[1] - camera_offset[1])
        end_x = int(x + self.vel[0] * 0.5)
        end_y = int(y + self.vel[1] * 0.5)
        
        # Glow effect for projectile
        pygame.draw.line(surface, (255, 100, 100, 50), (x, y), (end_x, end_y), 8)
        pygame.draw.line(surface, (255, 200, 200, 200), (x, y), (end_x, end_y), 4)


class Racer:
    def __init__(self, pos, angle, color, is_player=False):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.angle = angle
        self.color = color
        self.is_player = is_player

        self.health = 100
        self.max_health = 100
        
        self.current_lap = 1
        self.next_checkpoint_idx = 1
        
        self.item = None # 'shield' or 'missile'
        self.shield_timer = 0
        
        self.boost_fuel = 100
        self.max_boost_fuel = 100
        self.is_boosting = False
        
        self.trail = []
        self.size = 12

    def update(self, movement, use_item, use_boost, track_checkpoints):
        # Physics constants
        turn_speed = 0.08
        acceleration = 0.3
        boost_acceleration = 0.7
        friction = 0.97
        max_speed = 12.0
        
        # Handle turning
        if movement == 3: # Left
            self.angle -= turn_speed
        if movement == 4: # Right
            self.angle += turn_speed

        # Handle acceleration and braking
        self.is_boosting = False
        direction_vector = np.array([math.cos(self.angle), math.sin(self.angle)])
        if movement == 1: # Accelerate
            self.vel += direction_vector * acceleration
        if movement == 2: # Brake
            self.vel *= 0.92

        # Handle boost
        if use_boost and self.boost_fuel > 0:
            self.vel += direction_vector * boost_acceleration
            self.boost_fuel = max(0, self.boost_fuel - 2)
            self.is_boosting = True
        else:
            self.boost_fuel = min(self.max_boost_fuel, self.boost_fuel + 0.2)
            
        # Apply friction and speed cap
        speed = np.linalg.norm(self.vel)
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed
        self.vel *= friction
        
        # Update position
        self.pos += self.vel

        # Update trail
        self.trail.append(tuple(self.pos))
        if len(self.trail) > 20:
            self.trail.pop(0)

        # Update shield
        if self.shield_timer > 0:
            self.shield_timer -= 1
            
        # Item usage is handled in the main env loop to spawn projectiles
        
    def get_rotated_points(self):
        points = [
            (-self.size, -self.size * 0.7),
            (self.size * 1.5, 0),
            (-self.size, self.size * 0.7)
        ]
        rotated_points = []
        for x, y in points:
            new_x = x * math.cos(self.angle) - y * math.sin(self.angle) + self.pos[0]
            new_y = x * math.sin(self.angle) + y * math.cos(self.angle) + self.pos[1]
            rotated_points.append((new_x, new_y))
        return rotated_points

    def draw(self, surface, camera_offset):
        # Trail
        if len(self.trail) > 2:
            screen_trail = [(p[0] - camera_offset[0], p[1] - camera_offset[1]) for p in self.trail]
            trail_color = (*self.color, 150) if not self.is_boosting else (255, 255, 150, 200)
            trail_width = 3 if not self.is_boosting else 6
            pygame.draw.lines(surface, trail_color, False, screen_trail, trail_width)

        # Shield effect
        if self.shield_timer > 0:
            shield_alpha = 50 + (self.shield_timer % 10) * 10
            x, y = int(self.pos[0] - camera_offset[0]), int(self.pos[1] - camera_offset[1])
            pygame.gfxdraw.aacircle(surface, x, y, self.size + 10, (150, 200, 255, shield_alpha))
            pygame.gfxdraw.filled_circle(surface, x, y, self.size + 10, (150, 200, 255, shield_alpha // 2))

        # Racer body
        poly_points = self.get_rotated_points()
        screen_points = [(p[0] - camera_offset[0], p[1] - camera_offset[1]) for p in poly_points]
        
        # Glow
        glow_color = (*self.color, 50)
        x, y = int(self.pos[0] - camera_offset[0]), int(self.pos[1] - camera_offset[1])
        pygame.gfxdraw.filled_circle(surface, x, y, self.size + 5, glow_color)
        
        # Main body
        pygame.gfxdraw.aapolygon(surface, screen_points, self.color)
        pygame.gfxdraw.filled_polygon(surface, screen_points, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Futuristic hover-racer on neon tracks. Use boosts, pick up items like missiles and "
        "shields, and outpace your opponents to win."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move, brake, and turn. Hold Shift to boost and press "
        "Space to use your current item."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 2400, 1600
        self.MAX_STEPS = 2500
        self.NUM_LAPS = 3
        self.NUM_AIS = 3

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_AI = [(255, 80, 80), (80, 150, 255), (255, 255, 80)]
        self.COLOR_TRACK = (50, 30, 80)
        self.COLOR_CHECKPOINT = (200, 200, 255)
        self.COLOR_ITEM_OFFENSIVE = (255, 50, 50)
        self.COLOR_ITEM_DEFENSIVE = (50, 150, 255)
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.player = None
        self.ais = []
        self.projectiles = []
        self.particles = []
        self.item_spawns = []
        self.track_checkpoints = []

        self.player_prev_space_held = False
        
        # self.reset() is called by the wrapper or user
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None
        
        self.projectiles.clear()
        self.particles.clear()
        
        self._generate_track()
        self._initialize_entities()

        self.player_prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input & Update Player ---
        movement, space_held_int, shift_held_int = action
        space_pressed = space_held_int == 1 and not self.player_prev_space_held
        shift_held = shift_held_int == 1
        self.player_prev_space_held = (space_held_int == 1)

        # Track progress for reward
        dist_before = np.linalg.norm(self.player.pos - self.track_checkpoints[self.player.next_checkpoint_idx])
        
        self.player.update(movement, space_pressed, shift_held, self.track_checkpoints)
        self._constrain_racer(self.player)

        dist_after = np.linalg.norm(self.player.pos - self.track_checkpoints[self.player.next_checkpoint_idx])
        reward += (dist_before - dist_after) * 0.05 # Reward for getting closer to checkpoint

        # Use item
        if space_pressed and self.player.item:
            if self.player.item == 'missile':
                # Sfx: missile_fire.wav
                self.projectiles.append(Projectile(self.player.pos + np.array([math.cos(self.player.angle), math.sin(self.player.angle)]) * 20, self.player.angle, self.player))
            elif self.player.item == 'shield':
                # Sfx: shield_activate.wav
                self.player.shield_timer = 150 # 5 seconds
            self.player.item = None

        # --- Update AI ---
        for ai in self.ais:
            self._update_ai(ai)
            self._constrain_racer(ai)

        # --- Update Game Objects ---
        self.projectiles = [p for p in self.projectiles if p.update()]
        self.particles = [p for p in self.particles if p.update()]
        
        # --- Handle Collisions & Items ---
        reward += self._handle_collisions()
        reward += self._handle_item_collection()
        
        # --- Update Laps ---
        reward += self._update_lap(self.player)
        for ai in self.ais:
            self._update_lap(ai)

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.winner is self.player:
                reward += 100
                self.score += 100
            elif self.player.health <= 0:
                reward -= 100
                self.score -= 100
        
        self.score += reward
        
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _generate_track(self):
        self.track_checkpoints.clear()
        center_x, center_y = self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2
        num_points = 12
        base_radius = min(self.WORLD_WIDTH, self.WORLD_HEIGHT) * 0.4
        
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = base_radius + self.np_random.uniform(-base_radius * 0.2, base_radius * 0.2)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append(np.array([x, y]))
        
        self.track_checkpoints = points

    def _initialize_entities(self):
        start_pos = self.track_checkpoints[0] + np.array([0, 50])
        start_angle = self._get_angle_to_point(start_pos, self.track_checkpoints[1])

        self.player = Racer(start_pos, start_angle, self.COLOR_PLAYER, is_player=True)
        
        self.ais.clear()
        for i in range(self.NUM_AIS):
            offset = np.array([(i % 2 - 0.5) * 80, (i // 2 + 1) * 60])
            ai_pos = start_pos + offset
            ai = Racer(ai_pos, start_angle, self.COLOR_AI[i])
            self.ais.append(ai)
        
        self.item_spawns.clear()
        for i in range(len(self.track_checkpoints)):
            if i % 2 == 1:
                p1 = self.track_checkpoints[i]
                p2 = self.track_checkpoints[(i + 1) % len(self.track_checkpoints)]
                midpoint = (p1 + p2) / 2
                self.item_spawns.append({'pos': midpoint, 'radius': 20, 'collected_timer': 0})
    
    def _update_ai(self, ai):
        target_pos = self.track_checkpoints[ai.next_checkpoint_idx]
        
        # Basic navigation
        target_angle = self._get_angle_to_point(ai.pos, target_pos)
        angle_diff = (target_angle - ai.angle + math.pi) % (2 * math.pi) - math.pi
        
        movement = 1 # Accelerate
        if abs(angle_diff) > 0.2:
            movement = 3 if angle_diff < 0 else 4 # Turn
        
        # Simple item usage
        use_item = False
        if ai.item and self.np_random.random() < 0.05:
            use_item = True
            if ai.item == 'missile':
                self.projectiles.append(Projectile(ai.pos, ai.angle, ai))
            elif ai.item == 'shield':
                ai.shield_timer = 150
            ai.item = None
        
        ai.update(movement, use_item, False, self.track_checkpoints)

    def _get_angle_to_point(self, pos, target_pos):
        return math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0])

    def _constrain_racer(self, racer):
        # Bounce off world boundaries
        reward_penalty = 0
        if racer.pos[0] < racer.size:
            racer.pos[0] = racer.size
            racer.vel[0] *= -0.5
            reward_penalty = -0.01
        elif racer.pos[0] > self.WORLD_WIDTH - racer.size:
            racer.pos[0] = self.WORLD_WIDTH - racer.size
            racer.vel[0] *= -0.5
            reward_penalty = -0.01
        if racer.pos[1] < racer.size:
            racer.pos[1] = racer.size
            racer.vel[1] *= -0.5
            reward_penalty = -0.01
        elif racer.pos[1] > self.WORLD_HEIGHT - racer.size:
            racer.pos[1] = self.WORLD_HEIGHT - racer.size
            racer.vel[1] *= -0.5
            reward_penalty = -0.01
        if racer.is_player and reward_penalty < 0:
            # Sfx: wall_thud.wav
            self.score += reward_penalty

    def _handle_collisions(self):
        reward = 0
        all_racers = [self.player] + self.ais
        
        # Projectile-Racer collisions
        for proj in self.projectiles[:]:
            for racer in all_racers:
                if proj.owner is racer: continue
                dist = np.linalg.norm(proj.pos - racer.pos)
                if dist < racer.size + proj.radius:
                    if racer.shield_timer > 0:
                        # Sfx: shield_deflect.wav
                        if proj.owner is self.player: reward += 2
                        racer.shield_timer = 0 # Shield breaks on hit
                    else:
                        # Sfx: explosion.wav
                        racer.health = max(0, racer.health - 25)
                        if proj.owner is self.player: reward += 5
                        for _ in range(30):
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            speed = self.np_random.uniform(1, 5)
                            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                            self.particles.append(Particle(racer.pos, vel, racer.color, self.np_random.integers(2, 7), 30))
                    
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    break
        return reward

    def _handle_item_collection(self):
        reward = 0
        all_racers = [self.player] + self.ais
        for spawn in self.item_spawns:
            if spawn['collected_timer'] > 0:
                spawn['collected_timer'] -= 1
                continue
            
            for racer in all_racers:
                if racer.item is None:
                    dist = np.linalg.norm(spawn['pos'] - racer.pos)
                    if dist < spawn['radius'] + racer.size:
                        # Sfx: item_pickup.wav
                        racer.item = self.np_random.choice(['missile', 'shield'])
                        spawn['collected_timer'] = 300 # 10 seconds respawn
                        if racer.is_player: reward += 1
                        break
        return reward

    def _update_lap(self, racer):
        if racer.current_lap > self.NUM_LAPS: return 0
        
        checkpoint_pos = self.track_checkpoints[racer.next_checkpoint_idx]
        dist = np.linalg.norm(racer.pos - checkpoint_pos)
        
        if dist < 60: # Checkpoint radius
            racer.next_checkpoint_idx += 1
            if racer.next_checkpoint_idx >= len(self.track_checkpoints):
                racer.next_checkpoint_idx = 0
                racer.current_lap += 1
                if racer.is_player:
                    # Sfx: lap_complete.wav
                    pass
        return 0
    
    def _check_termination(self):
        if self.player.current_lap > self.NUM_LAPS:
            self.winner = self.player
            return True
        
        for ai in self.ais:
            if ai.current_lap > self.NUM_LAPS:
                self.winner = ai
                return True

        if self.player.health <= 0:
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        camera_offset = self.player.pos - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        
        self._render_game(camera_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, camera_offset):
        # Draw track
        screen_points = [ (p[0]-camera_offset[0], p[1]-camera_offset[1]) for p in self.track_checkpoints]
        pygame.draw.lines(self.screen, self.COLOR_TRACK, True, screen_points, 30)
        
        # Draw checkpoints
        next_cp_idx = self.player.next_checkpoint_idx
        cp_pos = self.track_checkpoints[next_cp_idx]
        x, y = int(cp_pos[0] - camera_offset[0]), int(cp_pos[1] - camera_offset[1])
        alpha = 100 + (self.steps % 30) * 3
        pygame.gfxdraw.aacircle(self.screen, x, y, 60, (*self.COLOR_CHECKPOINT, alpha))

        # Draw item spawns
        for spawn in self.item_spawns:
            x, y = int(spawn['pos'][0] - camera_offset[0]), int(spawn['pos'][1] - camera_offset[1])
            if spawn['collected_timer'] == 0:
                alpha = 150 + (self.steps % 20) * 5
                item_type = 'missile' if (spawn['pos'][0] % 100) > 50 else 'shield'
                color = self.COLOR_ITEM_OFFENSIVE if item_type == 'missile' else self.COLOR_ITEM_DEFENSIVE
                pygame.gfxdraw.aacircle(self.screen, x, y, spawn['radius'], (*color, alpha))
                pygame.gfxdraw.filled_circle(self.screen, x, y, spawn['radius'], (*color, alpha // 2))

        # Draw entities
        all_entities = self.ais + [self.player] + self.projectiles + self.particles
        for entity in sorted(all_entities, key=lambda e: e.pos[1] if hasattr(e, 'pos') else 0):
             entity.draw(self.screen, camera_offset)

    def _render_ui(self):
        # Lap counter
        lap_text = self.font_large.render(f"LAP: {min(self.player.current_lap, self.NUM_LAPS)} / {self.NUM_LAPS}", True, (255, 255, 255))
        self.screen.blit(lap_text, (10, 10))

        # Health bar
        health_pct = self.player.health / self.player.max_health
        pygame.draw.rect(self.screen, (80, 0, 0), (10, self.SCREEN_HEIGHT - 30, 200, 20))
        pygame.draw.rect(self.screen, (0, 180, 0), (10, self.SCREEN_HEIGHT - 30, 200 * health_pct, 20))
        
        # Boost bar
        boost_pct = self.player.boost_fuel / self.player.max_boost_fuel
        pygame.draw.rect(self.screen, (80, 80, 0), (10, self.SCREEN_HEIGHT - 55, 200, 20))
        pygame.draw.rect(self.screen, (255, 200, 0), (10, self.SCREEN_HEIGHT - 55, 200 * boost_pct, 20))

        # Item indicator
        if self.player.item:
            color = self.COLOR_ITEM_OFFENSIVE if self.player.item == 'missile' else self.COLOR_ITEM_DEFENSIVE
            pygame.draw.rect(self.screen, color, (self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 60, 50, 50))
            item_text = self.font_small.render(self.player.item.upper(), True, (255,255,255))
            self.screen.blit(item_text, (self.SCREEN_WIDTH - 58, self.SCREEN_HEIGHT - 48))

        # Minimap
        self._render_minimap()

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.winner is self.player else "GAME OVER"
            text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _render_minimap(self):
        map_w, map_h = 120, 80
        map_x, map_y = self.SCREEN_WIDTH - map_w - 10, 10
        map_surf = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        map_surf.fill((0, 0, 0, 100))
        
        scale_x = map_w / self.WORLD_WIDTH
        scale_y = map_h / self.WORLD_HEIGHT

        map_points = [(p[0] * scale_x, p[1] * scale_y) for p in self.track_checkpoints]
        pygame.draw.lines(map_surf, (*self.COLOR_TRACK, 200), True, map_points, 2)
        
        for racer in [self.player] + self.ais:
            rx, ry = int(racer.pos[0] * scale_x), int(racer.pos[1] * scale_y)
            pygame.draw.circle(map_surf, racer.color, (rx, ry), 3)
            
        self.screen.blit(map_surf, (map_x, map_y))
        pygame.draw.rect(self.screen, (255, 255, 255, 100), (map_x, map_y, map_w, map_h), 1)

    def _get_info(self):
        winner_status = "none"
        if self.winner:
            winner_status = "player" if self.winner is self.player else "ai"
        
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": min(self.player.current_lap, self.NUM_LAPS + 1),
            "health": self.player.health,
            "winner": winner_status
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the autograder.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Astral Arena Racer")
    clock = pygame.time.Clock()
    
    movement = 0
    space = 0
    shift = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS
        print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Lap: {info['lap']}, Terminated: {terminated}, Truncated: {truncated}")

    env.close()