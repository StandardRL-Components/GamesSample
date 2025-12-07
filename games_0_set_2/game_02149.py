
# Generated: 2025-08-28T03:51:42.743688
# Source Brief: brief_02149.md
# Brief Index: 2149

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon (boost)."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_TRACK = (70, 80, 90)
        self.COLOR_OFFROAD = (40, 50, 60)
        self.COLOR_FINISH_LINE = (255, 255, 0)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_AI_1 = (255, 50, 50)
        self.COLOR_AI_2 = (50, 150, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_BOOST_BAR = (0, 200, 255)
        
        # Game constants
        self.NUM_LAPS = 2
        self.MAX_STEPS = 2500
        self.TRACK_WIDTH = 90
        self.NUM_WAYPOINTS = 20
        self.TRACK_RADIUS = 800

        # State variables will be initialized in reset()
        self.track_waypoints = []
        self.player = None
        self.opponents = []
        self.all_karts = []
        self.skidmarks = []
        self.particles = []
        self.steps = 0
        self.game_over = False
        self.game_over_timer = 0
        self.final_ranks = []
        self.last_player_rank = 1

        self.reset()
        
        # Run validation check
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self._generate_track()

        start_pos = self.track_waypoints[0]
        start_angle = self._get_waypoint_angle(0)

        self.player = Kart(pos=start_pos, angle=start_angle, color=self.COLOR_PLAYER, is_player=True)
        self.opponents = [
            Kart(pos=start_pos + pygame.Vector2(0, 30), angle=start_angle, color=self.COLOR_AI_1),
            Kart(pos=start_pos + pygame.Vector2(0, -30), angle=start_angle, color=self.COLOR_AI_2)
        ]
        self.all_karts = [self.player] + self.opponents

        for kart in self.all_karts:
            kart.reset(pos=kart.pos, angle=start_angle)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_timer = 0
        self.final_ranks = []
        self.last_player_rank = self._get_player_rank()
        
        self.skidmarks = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self._update_player(movement, space_held, shift_held)
            self._update_opponents()

            for kart in self.all_karts:
                kart.update(self.track_waypoints, self.TRACK_WIDTH)
                self._update_lap_counter(kart)

            self._update_effects(shift_held)
            reward = self._calculate_reward()
            self.score += reward
        else:
            self.game_over_timer += 1

        self.steps += 1
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_track(self):
        self.track_waypoints = []
        center_x, center_y = 0, 0
        
        base_angle = 2 * math.pi / self.NUM_WAYPOINTS
        for i in range(self.NUM_WAYPOINTS):
            angle = base_angle * i
            rand_radius = self.TRACK_RADIUS + self.np_random.uniform(-0.2, 0.2) * self.TRACK_RADIUS
            
            x = center_x + math.cos(angle) * rand_radius
            y = center_y + math.sin(angle) * rand_radius
            self.track_waypoints.append(pygame.Vector2(x, y))

    def _get_waypoint_angle(self, index):
        p1 = self.track_waypoints[index]
        p2 = self.track_waypoints[(index + 1) % len(self.track_waypoints)]
        return math.atan2(p2.y - p1.y, p2.x - p1.x)

    def _update_player(self, movement, space_held, shift_held):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        is_turning = movement in [3, 4]
        self.player.is_drifting = shift_held and is_turning and self.player.speed > 2.0

        if movement == 1: self.player.accelerate()
        if movement == 2: self.player.brake()
        if movement == 3: self.player.turn(-1)
        if movement == 4: self.player.turn(1)
        if space_held: self.player.boost() # sound: boost.wav

    def _update_opponents(self):
        for ai in self.opponents:
            target_point = self.track_waypoints[ai.next_waypoint_idx]
            
            # Add some wobble for realism
            target_point += pygame.Vector2(self.np_random.uniform(-15, 15), self.np_random.uniform(-15, 15))

            vector_to_target = target_point - ai.pos
            angle_to_target = math.atan2(vector_to_target.y, vector_to_target.x)
            
            angle_diff = (angle_to_target - ai.angle + math.pi) % (2 * math.pi) - math.pi
            
            if angle_diff > 0.1: ai.turn(1)
            elif angle_diff < -0.1: ai.turn(-1)
            
            ai.accelerate() # AI always tries to accelerate

    def _update_lap_counter(self, kart):
        if kart.lap >= self.NUM_LAPS:
            return

        target_wp_pos = self.track_waypoints[kart.next_waypoint_idx]
        if kart.pos.distance_to(target_wp_pos) < self.TRACK_WIDTH * 1.5:
            kart.next_waypoint_idx = (kart.next_waypoint_idx + 1)
            
            if kart.next_waypoint_idx == len(self.track_waypoints): # Passed last waypoint
                kart.next_waypoint_idx = 0
            
            if kart.next_waypoint_idx == 1 and kart.prev_waypoint_idx == 0: # Just crossed finish line
                kart.lap += 1
                if kart.is_player:
                    self.score += 1.0 # Event-based reward for lap completion
            
            kart.prev_waypoint_idx = (kart.next_waypoint_idx -1 + len(self.track_waypoints)) % len(self.track_waypoints)

    def _update_effects(self, shift_held):
        # Skid marks
        if self.player.is_drifting:
            self.skidmarks.append(Skidmark(self.player.pos, self.player.angle))
            # sound: drift.wav
        
        self.skidmarks = [s for s in self.skidmarks if s.update()]

        # Boost particles
        if self.player.is_boosting:
            for _ in range(3):
                self.particles.append(Particle(self.player.pos, self.player.angle, self.np_random))
        
        self.particles = [p for p in self.particles if p.update()]

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for rank change
        current_rank = self._get_player_rank()
        if current_rank < self.last_player_rank:
            reward += 0.1
        elif current_rank > self.last_player_rank:
            reward -= 0.1
        self.last_player_rank = current_rank

        # Reward for being on track
        if not self.player.is_offroad:
            reward += 0.01
        else:
            reward -= 0.05 # Penalty for being off-road
            
        return reward

    def _check_termination(self):
        if self.game_over:
            return self.game_over_timer > 90 # Wait 3 seconds after race end

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self._finalize_race()
            return False

        racers_finished = sum(1 for k in self.all_karts if k.lap >= self.NUM_LAPS)
        if racers_finished == len(self.all_karts):
            self.game_over = True
            self._finalize_race()
            # sound: race_finish.wav
            
            # Terminal reward
            final_rank = self.final_ranks.index(self.player) + 1
            if final_rank == 1:
                self.score += 100
            elif final_rank == 3:
                self.score -= 100
            
            return False # Let the game run for a few more frames to show results
        
        return False

    def _finalize_race(self):
        if not self.final_ranks:
            self.final_ranks = self._get_ranks()

    def _get_ranks(self):
        return sorted(self.all_karts, key=lambda k: (-k.lap, -k.next_waypoint_idx, k.pos.distance_to(self.track_waypoints[k.next_waypoint_idx])))

    def _get_player_rank(self):
        ranks = self._get_ranks()
        return ranks.index(self.player) + 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_OFFROAD)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.player.lap,
            "rank": self._get_player_rank() if not self.game_over else self.final_ranks.index(self.player) + 1,
        }

    def _world_to_screen(self, pos):
        return pos - self.player.pos + pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)

    def _render_game(self):
        # Render track
        track_points_screen = [self._world_to_screen(p) for p in self.track_waypoints]
        pygame.draw.polygon(self.screen, self.COLOR_TRACK, track_points_screen, self.TRACK_WIDTH)
        for p in track_points_screen:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), self.TRACK_WIDTH // 2, self.COLOR_TRACK)
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), self.TRACK_WIDTH // 2, self.COLOR_TRACK)

        # Render finish line
        p1 = self.track_waypoints[-1]
        p2 = self.track_waypoints[0]
        mid = (p1 + p2) / 2
        direction = (p2 - p1).normalize()
        perp_dir = pygame.Vector2(-direction.y, direction.x)
        
        for i in range(-4, 5, 2):
            start = self._world_to_screen(mid + perp_dir * i * 10)
            end = self._world_to_screen(mid + perp_dir * (i + 1) * 10)
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, start, end, 5)

        # Render skidmarks and particles (behind karts)
        for skid in self.skidmarks:
            skid.draw(self.screen, self._world_to_screen)
        for particle in self.particles:
            particle.draw(self.screen, self._world_to_screen)

        # Render karts
        for kart in sorted(self.all_karts, key=lambda k: k.pos.y):
             kart.draw(self.screen, self._world_to_screen)
    
    def _render_ui(self):
        # Lap counter
        lap_text = self.font_small.render(f"LAP: {min(self.player.lap + 1, self.NUM_LAPS)} / {self.NUM_LAPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))
        
        # Position
        rank = self._get_player_rank() if not self.game_over else self.final_ranks.index(self.player) + 1
        rank_str = {1: "1ST", 2: "2ND", 3: "3RD"}.get(rank, f"{rank}TH")
        rank_text = self.font_large.render(rank_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(rank_text, (self.WIDTH - rank_text.get_width() - 10, 10))

        # Boost Meter
        boost_width = 150
        boost_height = 15
        boost_pct = self.player.boost_charge / self.player.MAX_BOOST
        pygame.draw.rect(self.screen, (50,50,50), (10, self.HEIGHT - boost_height - 10, boost_width, boost_height))
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (10, self.HEIGHT - boost_height - 10, boost_width * boost_pct, boost_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, self.HEIGHT - boost_height - 10, boost_width, boost_height), 1)

        # Speed Lines for high speed
        if self.player.speed > 8:
            for _ in range(int(self.player.speed) - 7):
                angle = self.np_random.uniform(0, 2 * math.pi)
                length = self.np_random.uniform(100, 300)
                start = pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)
                end = start + pygame.Vector2(math.cos(angle), math.sin(angle)) * length
                alpha = int(100 * (self.player.speed - 8) / 5)
                color = (200, 200, 220, alpha)
                pygame.draw.line(self.screen, color, start, end, 1)

        # Game Over Text
        if self.game_over and self.game_over_timer > 30:
            final_rank = self.final_ranks.index(self.player) + 1
            msg = {1: "YOU WIN!", 2: "2ND PLACE", 3: "3RD PLACE"}[final_rank]
            color = {1: (255, 215, 0), 2: (192, 192, 192), 3: (205, 127, 50)}[final_rank]
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("Running implementation validation...")
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

class Kart:
    def __init__(self, pos, angle, color, is_player=False):
        self.is_player = is_player
        self.color = color
        self.chassis_color = tuple(c * 0.7 for c in color)
        self.reset(pos, angle)

    def reset(self, pos, angle):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.angle = angle
        self.speed = 0
        
        self.lap = 0
        self.next_waypoint_idx = 1
        self.prev_waypoint_idx = 0
        
        self.is_offroad = False
        self.is_drifting = False
        self.is_boosting = False
        
        # Player specific
        self.boost_charge = 100
        self.MAX_BOOST = 100
        
        # Physics constants
        self.MAX_SPEED = 12
        self.ACCELERATION = 0.2
        self.BRAKING = 0.4
        self.FRICTION = 0.08
        self.TURN_SPEED = 0.08
        self.DRIFT_TURN_MOD = 1.8
        self.DRIFT_FRICTION_MOD = 0.5
        self.OFFROAD_FRICTION_MOD = 5.0

    def accelerate(self):
        self.speed += self.ACCELERATION
        if self.is_drifting: self.speed = max(1.0, self.speed)

    def brake(self):
        self.speed -= self.BRAKING

    def turn(self, direction): # -1 for left, 1 for right
        turn_mod = self.DRIFT_TURN_MOD if self.is_drifting else 1.0
        self.angle += direction * self.TURN_SPEED * turn_mod * (1 - self.speed / (self.MAX_SPEED * 2))

    def boost(self):
        if self.boost_charge > 0:
            self.is_boosting = True
            self.speed += 0.8
            self.boost_charge -= 3
            # sound: boost_active.wav
        else:
            self.is_boosting = False

    def update(self, waypoints, track_width):
        # Recharge boost
        if not self.is_boosting:
            self.boost_charge = min(self.MAX_BOOST, self.boost_charge + 0.3)
        self.is_boosting = False # Reset every frame, only true if action is taken

        # Check offroad status
        closest_dist = float('inf')
        for i in range(len(waypoints)):
            p1 = waypoints[i]
            p2 = waypoints[(i + 1) % len(waypoints)]
            dist = self._point_segment_distance(self.pos, p1, p2)
            if dist < closest_dist:
                closest_dist = dist
        self.is_offroad = closest_dist > track_width / 2

        # Apply friction
        friction_mod = 1.0
        if self.is_drifting: friction_mod *= self.DRIFT_FRICTION_MOD
        if self.is_offroad: friction_mod *= self.OFFROAD_FRICTION_MOD
        self.speed -= self.FRICTION * friction_mod
        self.speed = max(0, min(self.speed, self.MAX_SPEED))

        # Update velocity vector
        heading_vector = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
        
        # Drifting physics
        if self.is_drifting:
            # Interpolate velocity towards heading slower for slide effect
            self.vel = self.vel.lerp(heading_vector * self.speed, 0.1)
        else:
            self.vel = heading_vector * self.speed
        
        self.pos += self.vel

    def draw(self, surface, world_to_screen):
        screen_pos = world_to_screen(self.pos)
        
        # Kart body
        kart_size = (30, 18)
        kart_surf = pygame.Surface(kart_size, pygame.SRCALPHA)
        
        # Main chassis
        pygame.draw.rect(kart_surf, self.chassis_color, (0, 0, kart_size[0], kart_size[1]), border_radius=4)
        # Cabin
        pygame.draw.rect(kart_surf, self.color, (8, 3, 18, 12), border_radius=3)
        
        rotated_kart = pygame.transform.rotate(kart_surf, -math.degrees(self.angle))
        rect = rotated_kart.get_rect(center=screen_pos)
        
        surface.blit(rotated_kart, rect.topleft)

        # Drifting glow
        if self.is_drifting:
            glow_radius = 20
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.02)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.color, int(alpha)))
            surface.blit(glow_surf, (screen_pos.x - glow_radius, screen_pos.y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _point_segment_distance(self, p, a, b):
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return p.distance_to(projection)

class Skidmark:
    def __init__(self, pos, angle):
        self.pos = pos.copy()
        self.angle = angle
        self.lifetime = 60
        self.alpha = 150
        self.width = 8
        self.length = 10

    def update(self):
        self.lifetime -= 1
        self.alpha = max(0, 150 * (self.lifetime / 60))
        return self.lifetime > 0

    def draw(self, surface, world_to_screen):
        screen_pos = world_to_screen(self.pos)
        
        skid_surf = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        skid_surf.fill((20, 20, 20, self.alpha))
        
        rotated_skid = pygame.transform.rotate(skid_surf, -math.degrees(self.angle))
        rect = rotated_skid.get_rect(center=screen_pos)
        surface.blit(rotated_skid, rect.topleft)


class Particle:
    def __init__(self, pos, angle, np_random):
        self.pos = pos.copy()
        self.vel = pygame.Vector2(np_random.uniform(-1, 1), np_random.uniform(-1, 1))
        # Eject particles opposite to kart direction
        self.vel -= pygame.Vector2(math.cos(angle), math.sin(angle)) * np_random.uniform(2, 4)
        self.lifetime = 20
        self.radius = np_random.uniform(2, 5)
        self.color = random.choice([(0, 200, 255), (100, 220, 255), (200, 255, 255)])

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifetime -= 1
        self.radius -= 0.1
        return self.lifetime > 0 and self.radius > 0

    def draw(self, surface, world_to_screen):
        screen_pos = world_to_screen(self.pos)
        alpha = 255 * (self.lifetime / 20)
        color = (*self.color, int(alpha))
        pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), int(self.radius), color)